"""
Cotton irrigation water consumption: ML prediction pipeline
=============================================================

Standalone (non-Colab) version of `4_cotton_prediction_cmip_v5.ipynb`.

What it does
------------
1. Loads observed water-consumption data and CMIP weather data (historical +
   SSP1/SSP2/SSP3/SSP5) from local files.
2. Builds monthly weather features per state/year and picks the most
   informative ones (correlation + Random Forest importance).
3. Tries several regression models with leave-year-out cross-validation,
   scored on within-state R^2 (i.e. against each state's own mean -- this is
   the metric a reviewer cares about, not the pooled R^2 across states).
4. Validates the selected model per state, year-by-year, against naive
   baselines (state mean / persistence / linear trend), and saves a figure +
   a metrics table.
5. Refits the selected model on all historical years (2000-2023) and
   produces future (2024-2049) projections for each SSP scenario, saved to
   an Excel workbook.

How to run it
--------------
1. Put this script in the same folder as your data files:
     - water consumption.xlsx
     - abandonment_rate.xlsx        (loaded for completeness; not used below)
     - era5_features_2000_2023.csv  (real observed weather; run
                                      fetch_era5_weather.py first to build it)
     - results_SSP1_8f.csv          (future 2024-2049 only)
     - results_SSP2_8f.csv
     - results_SSP3_8f.csv
     - results_SSP5_8f.csv
   (Or edit DATA_DIR below to point at wherever they live.)

2. Install dependencies (a requirements.txt is provided alongside this
   script):
     pip install -r requirements.txt
   xgboost, lightgbm and tabpfn are optional extras -- the script skips them
   automatically if they aren't installed.

3. Run it:
     python cotton_water_prediction.py

Outputs are written to an "outputs" folder created next to this script:
   outputs/figures/figure_per_state_validation.tiff
   outputs/figures/figure_scatter_plot.tiff
   outputs/tables/per_state_validation_metrics.xlsx
   outputs/tables/future_water_predictions.xlsx   (one sheet per SSP scenario)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import pearsonr

from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.base import clone

# ============================================================================
# CONFIG -- edit these paths if your files live somewhere else
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATA_DIR = REPO_ROOT / "results" / "processed_data and results"
WEATHER_DIR = REPO_ROOT / "data" / "Weather"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Optional: a folder of .ttf font files if you want Arial styling exactly as
# in the original notebook. If it doesn't exist, matplotlib's default font is
# used instead and the script still runs fine.
FONT_DIR = SCRIPT_DIR / "fonts"

WATER_XLSX = DATA_DIR / "water consumption.xlsx"
ABANDONMENT_XLSX = DATA_DIR / "abandonment_rate.xlsx"
# Real observed weather (ERA5/ERA5-Land reanalysis, see fetch_era5_weather.py)
# for 2000-2023 -- used for BOTH training (2000-2014) and validation
# (2015-2023). The CMIP6 SSP files are free-running GCM output, so "year N"
# in them doesn't correspond to the real year N; they're only valid for
# genuinely-future years where no observations can exist.
ERA5_FEATURES_CSV = WEATHER_DIR / "era5_features_2000_2023.csv"
# Used ONLY as a bias-correction reference (see bias_correct_future_features)
# -- not as training data, since it's the free-running GCM run, not real
# weather. Its own 2000-2014 climatology tells us how that GCM's baseline
# differs from real (ERA5) conditions, so we can correct its 2024-2049
# projections onto an ERA5-consistent scale before the model ever sees them.
WEATHER_HISTORICAL_CSV = WEATHER_DIR / "CMIP" / "historical" / "results_his_8f.csv"
WEATHER_SSP_CSVS = {
    "SSP1": WEATHER_DIR / "CMIP" / "SSP1" / "results_SSP1_8f.csv",
    "SSP2": WEATHER_DIR / "CMIP" / "SSP2" / "results_SSP2_8f.csv",
    "SSP3": WEATHER_DIR / "CMIP" / "SSP3" / "results_SSP3_8f.csv",
    "SSP5": WEATHER_DIR / "CMIP" / "SSP5" / "results_SSP5_8f.csv",
}

IMPORTANCE_THRESHOLD = 0.005

N_STATES = 17
N_TRAIN_YEARS = 15    # 2000-2014
N_TEST_YEARS = 9       # 2015-2023
N_PREDICT_YEARS = 26   # 2024-2049

STATE_NAMES = ['AL', 'AZ', 'AR', 'CA', 'FL', 'GA', 'KS', 'LA', 'MS', 'MO',
               'NM', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA']
SSP_LABELS = ['SSP1', 'SSP2', 'SSP3', 'SSP5']


def setup_plot_style():
    """Match the original figure styling; falls back gracefully if the
    custom font folder isn't present."""
    if FONT_DIR.exists():
        for font_file in font_manager.findSystemFonts(fontpaths=[str(FONT_DIR)]):
            font_manager.fontManager.addfont(font_file)
        matplotlib.rcParams['font.family'] = "Arial"
        matplotlib.rcParams['font.sans-serif'] = ["Arial"]
    else:
        print(f"No font folder at {FONT_DIR} -- using matplotlib's default font.")

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 8


def load_data():
    print("Loading input data...")
    data_water = pd.read_excel(WATER_XLSX, index_col=0)
    data_abandonment = pd.read_excel(ABANDONMENT_XLSX, index_col=0).iloc[:, ::-1]

    era5_features = pd.read_csv(ERA5_FEATURES_CSV, index_col=['state', 'year'])
    weather_ssp = {label: pd.read_csv(path) for label, path in WEATHER_SSP_CSVS.items()}

    assert data_water.shape[0] == N_STATES, (
        f"water consumption.xlsx has {data_water.shape[0]} rows, expected {N_STATES} "
        "(one per state). Check the file's index column."
    )
    assert data_water.shape[1] >= N_TRAIN_YEARS + N_TEST_YEARS, (
        f"water consumption.xlsx has {data_water.shape[1]} year columns, expected at "
        f"least {N_TRAIN_YEARS + N_TEST_YEARS} (2000-2023)."
    )
    print(f"  water consumption.xlsx state order: {list(data_water.index)}")
    print(f"  (expected to correspond, in order, to: {STATE_NAMES})")

    return data_water, data_abandonment, era5_features, weather_ssp


def process_weather_data(weather_data, total_month, start=0, end=None):
    """Process raw CMIP weather data into (state x feature) arrays."""
    weather_array = np.array(weather_data).reshape(8, 17, total_month)
    # Raw data is chronological (flat index = year*12 + month), so the year
    # axis must be split off first (size total_month//12) with month last
    # (size 12) -- NOT (12, total_month//12), which would scramble months
    # and years together since that assumes month-major storage.
    weather_array = weather_array.reshape(8, 17, total_month // 12, 12)
    weather_array = weather_array.transpose(1, 0, 3, 2).reshape(17, 96, total_month // 12)[:, :, start:end]

    # Reorder rows from the CMIP file's native state order into STATE_NAMES order
    new_indices = [9, 15, 16, 8, 5, 10, 13, 4, 0, 11, 6, 1, 2, 14, 12, 7, 3]
    weather_reordered = weather_array[new_indices]

    features_flattened = weather_reordered.transpose(0, 2, 1).reshape(-1, 96)
    return features_flattened


MONTH_LABELS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
                'OCT', 'NOV', 'DEC']
WEATHER_CATEGORIES = ['pr', 'huss', 'mrro', 'sfcWind', 'tas', 'clt', 'evspsbl', 'mrsos']


def load_era5_features(era5_features):
    """Reorder era5_features_2000_2023.csv (indexed by state, year) into the
    same state-major / year-ascending row order used by state_onehot() and
    data_water.ravel() (STATE_NAMES order), and split into the training
    (2000-2014) and validation (2015-2023) blocks."""
    ordered_cols = [f'{cat}_{month}' for cat in WEATHER_CATEGORIES for month in MONTH_LABELS]

    def year_block(year_start, year_end):
        rows = [era5_features.loc[(state, year), ordered_cols].values
                for state in STATE_NAMES for year in range(year_start, year_end + 1)]
        return np.array(rows, dtype=float)

    features_train = year_block(2000, 2014)
    features_val = year_block(2015, 2023)
    return features_train, features_val


# Categories where the CMIP GCM's own bias vs. ERA5 is corrected multiplicatively
# (non-negative, ratio-natured quantities -- standard climate bias-correction practice).
RATIO_CORRECTED_CATEGORIES = {'pr', 'huss', 'mrro', 'sfcWind', 'evspsbl', 'mrsos'}
# Categories corrected additively (temperature, cloud fraction).
ADDITIVE_CORRECTED_CATEGORIES = {'tas', 'clt'}


def bias_correct_future_features(features_predict, features_train_era5, weather_historical):
    """The CMIP GCM used for the SSP scenarios has its own climatological bias
    relative to real weather (e.g. its own mrro runs ~66% high, sfcWind ~4x
    high, vs. ERA5 -- confirmed by comparing the two over the 2000-2014
    overlap). Left uncorrected, the model -- trained entirely on ERA5-scale
    features -- would see systematically out-of-distribution feature values
    for every future (2024-2049) prediction. Standard fix: compute each
    feature's correction factor from the overlap period (ERA5 vs. this GCM's
    own "historical" run, both 2000-2014) and apply it to the future SSP data
    before the model ever sees it."""
    cmip_hist = process_weather_data(weather_historical, 180)  # 2000-2014, same rows/cols as features_train_era5

    era5_col_mean = features_train_era5.mean(axis=0)
    cmip_col_mean = cmip_hist.mean(axis=0)

    corrected = {}
    for label, F in features_predict.items():
        F = F.copy()
        for cat_idx, cat in enumerate(WEATHER_CATEGORIES):
            cols = slice(cat_idx * 12, cat_idx * 12 + 12)
            if cat in RATIO_CORRECTED_CATEGORIES:
                factor = era5_col_mean[cols] / cmip_col_mean[cols]
                F[:, cols] = np.clip(F[:, cols] * factor, 0, None)
            else:  # additive
                delta = era5_col_mean[cols] - cmip_col_mean[cols]
                F[:, cols] = F[:, cols] + delta
        corrected[label] = F
    # clt (cloud fraction) is physically bounded to [0, 100]
    for label in corrected:
        clt_cols = slice(WEATHER_CATEGORIES.index('clt') * 12, WEATHER_CATEGORIES.index('clt') * 12 + 12)
        corrected[label][:, clt_cols] = np.clip(corrected[label][:, clt_cols], 0, 100)
    return corrected


def build_features(era5_features, weather_ssp):
    print("Processing weather data...")
    features_train, features_val = load_era5_features(era5_features)

    # Future (2024-2049) must come from the SSP scenarios -- no observations
    # exist yet for those years -- but bias-correct them onto the ERA5 scale
    # the model was trained on (see bias_correct_future_features).
    features_predict_raw = {
        label: process_weather_data(weather_ssp[label], 420, start=9) for label in SSP_LABELS
    }
    weather_historical = pd.read_csv(WEATHER_HISTORICAL_CSV)
    features_predict = bias_correct_future_features(features_predict_raw, features_train, weather_historical)

    print(f"Training data shape: {features_train.shape}")
    print(f"Validation data shape: {features_val.shape}")

    return features_train, features_val, features_predict


def correlation_coefficients(X, y):
    return np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])


def select_important_features(features_train, water_targets_train):
    print("Analyzing feature importance...")
    X_train, y_train = features_train, water_targets_train

    correlation_importances = correlation_coefficients(X_train, y_train)

    rfc = RandomForestRegressor(n_estimators=1000, random_state=42)
    rfc.fit(X_train, y_train)
    rfc_importances = rfc.feature_importances_

    correlation_importances_abs = np.abs(correlation_importances) / np.sum(np.abs(correlation_importances))
    rfc_importances_norm = rfc_importances / np.sum(rfc_importances)
    overall_importance = np.mean([correlation_importances_abs, rfc_importances_norm], axis=0)

    features = [f'{category}_{month}' for category in WEATHER_CATEGORIES for month in MONTH_LABELS]

    importance_df = pd.DataFrame({
        'Feature': features,
        'Correlation Importance': correlation_importances,
        'RFC Importance': rfc_importances,
        'Overall Importance': overall_importance,
    })

    important_features = importance_df[importance_df['Overall Importance'] > IMPORTANCE_THRESHOLD]['Feature'].values
    important_indices = [features.index(f) for f in important_features]

    print(f"Selected {len(important_features)} important features")
    return important_indices


def state_onehot(n_years):
    sid = np.repeat(np.arange(N_STATES), n_years)
    oh = np.zeros((N_STATES * n_years, N_STATES))
    oh[np.arange(len(sid)), sid] = 1.0
    return oh


def fit_state_log_trends(dw, n_fit_years):
    """Fit each state's own log-linear trend (irrigation water use often
    changes by roughly a constant PERCENTAGE per year -- e.g. Ogallala-driven
    decline in Kansas -- and can't go negative, so log-space is the natural
    choice; a plain linear fit can overshoot into unrealistic/negative
    territory when extrapolated). Uses the first n_fit_years columns of dw
    (17, n_years_total). Returns (intercepts, slopes), each shape (17,)."""
    t = np.arange(n_fit_years)
    intercepts = np.zeros(N_STATES)
    slopes = np.zeros(N_STATES)
    for i in range(N_STATES):
        slope, intercept = np.polyfit(t, np.log(dw[i, :n_fit_years]), 1)
        intercepts[i], slopes[i] = intercept, slope
    return intercepts, slopes


def trend_predict(intercepts, slopes, year_indices):
    """year_indices: (n_years,), applied to every state alike. Returns
    level-space predictions, shape (17, n_years)."""
    t = np.asarray(year_indices)
    log_pred = intercepts[:, None] + slopes[:, None] * t[None, :]
    return np.exp(log_pred)


def residual_targets_log(dw, intercepts, slopes, year_start, n_years):
    """log(actual) - log(trend), flattened state-major/year-minor to match
    state_onehot()'s row order."""
    t = np.arange(year_start, year_start + n_years)
    pred = trend_predict(intercepts, slopes, t)
    actual = dw[:, year_start:year_start + n_years]
    return (np.log(actual) - np.log(pred)).ravel()


def within_state_r2(y_true, y_pred, state_ids):
    """R^2 against each state's OWN mean. 0.0 == exactly as good as always
    predicting that state's average -- the reviewer-relevant metric."""
    ss_res = ss_within = 0.0
    for s in np.unique(state_ids):
        m = state_ids == s
        ss_res += np.sum((y_true[m] - y_pred[m]) ** 2)
        ss_within += np.sum((y_true[m] - y_true[m].mean()) ** 2)
    return 1.0 - ss_res / ss_within


def leave_year_out_oof(model, X, y, year_ids, n_splits=5):
    """Out-of-fold predictions holding out YEARS (matches the real task)."""
    oof = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=year_ids):
        m = clone(model)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def build_candidate_models():
    candidates = {
        'Ridge (state FE)': RidgeCV(alphas=np.logspace(-3, 4, 30)),
        'RandomForest': RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                               random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                           random_state=42, n_jobs=-1),
        'HistGradBoost': HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                         max_leaf_nodes=15, l2_regularization=1.0,
                                                         early_stopping=False, random_state=42),
        # Unlike RandomForest/ExtraTrees/a KNN candidate would be, an MLP is a
        # smooth parametric function of its inputs, not an average over
        # training targets -- it isn't mathematically bounded to
        # [min(y_train), max(y_train)]. That doesn't matter much for the
        # residual predicted here (see fit_state_log_trends -- the trend
        # component is what needs to extrapolate for 2024-2049, and it
        # already does via ordinary log-linear extrapolation), but it's a
        # genuinely different model family from the tree-based candidates
        # above, worth having in the comparison on its own merits.
        'MLP': Pipeline([('scaler', StandardScaler()),
                          ('mlp', MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=2000,
                                                early_stopping=True, n_iter_no_change=20,
                                                random_state=0))]),
    }
    try:
        from xgboost import XGBRegressor
        candidates['XGBoost'] = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                              random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        print("  (xgboost not available - skipping)")
    try:
        from lightgbm import LGBMRegressor
        candidates['LightGBM'] = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15,
                                                min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                                                reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        print("  (lightgbm not available - skipping)")
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        gp_kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)
        candidates['GaussianProcess'] = Pipeline([
            ('scaler', StandardScaler()),
            ('gp', GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True,
                                             alpha=1e-6, random_state=42)),
        ])
    except ImportError:
        pass
    # TabPFN: transformer pre-trained specifically for small tabular datasets
    # (<10k rows) -- this regime (255 rows) is exactly what it targets.
    try:
        from tabpfn import TabPFNRegressor
        candidates['TabPFN'] = TabPFNRegressor(device='auto')
        print("  TabPFN available - included")
    except Exception:
        print("  (TabPFN not installed - run `pip install tabpfn` to include it)")

    return candidates


def select_model(X_train_m, y_train_m, state_ids_train, year_ids_train, enforce_nonnegative):
    """enforce_nonnegative: True when y_train_m is a level quantity that
    physically can't go negative (e.g. irrigation water use). False when
    y_train_m is a residual/log-residual, which is legitimately centered on
    zero and can go either way -- in that case a model that predicts
    negative values is completely normal, not a sign of a bad model."""
    candidates = build_candidate_models()

    print(f"\nEvaluating {len(candidates)} candidates with leave-year-out CV...")
    rows = []
    for name, model in candidates.items():
        try:
            oof = leave_year_out_oof(model, X_train_m, y_train_m, year_ids_train, n_splits=5)
            rows.append({
                'Model': name,
                'CV_within_state_R2': within_state_r2(y_train_m, oof, state_ids_train),
                'CV_pooled_R2': r2_score(y_train_m, oof),
                'CV_RMSE': np.sqrt(mean_squared_error(y_train_m, oof)),
                'min_pred': oof.min(),
            })
        except Exception as e:
            print(f"  {name} failed: {type(e).__name__}: {e}")

    selection_df = pd.DataFrame(rows).set_index('Model').sort_values('CV_within_state_R2', ascending=False)
    selection_df['nonneg_safe'] = selection_df['min_pred'] >= 0
    print("\n--- Model selection (ranked by within-state R2; 0.0 = state-average baseline) ---")
    print(selection_df.round(3))

    # Irrigation water cannot be negative. RandomForest/ExtraTrees average
    # training targets so they're bounded within [min(y), max(y)]; boosting
    # is additive and linear/GP models are unbounded, so they can undershoot
    # below zero, especially when extrapolating to future climate.
    eligible = selection_df[selection_df['nonneg_safe']] if enforce_nonnegative else selection_df
    if len(eligible) == 0:
        print("\nNo candidate stayed non-negative; falling back to full ranking with clipping at 0.")
        eligible = selection_df

    best_model_name = eligible.index[0]
    best_model = candidates[best_model_name]
    best_cv = eligible.iloc[0]['CV_within_state_R2']
    print(f"\nSelected: {best_model_name}  (CV within-state R2 = {best_cv:.3f})")

    excluded = selection_df[~selection_df['nonneg_safe']]
    if enforce_nonnegative and len(excluded) > 0:
        print(f"Excluded for producing negative predictions: {', '.join(excluded.index)}")

    if best_cv <= 0:
        print("\nWARNING: best CV within-state R2 is <= 0. No candidate beats simply predicting\n"
              "each state's own average on held-out years. The validation step below compares\n"
              "against naive baselines to show whether this is a modelling ceiling or a data ceiling.")

    return best_model_name, best_model


def validate_model(best_model_name, best_model, X_train_resid, resid_train_log, X_val_resid,
                    trend_intercepts, trend_slopes, data_water):
    print(f"\nValidating {best_model_name} (log-linear trend + weather-residual correction, "
          f"trend and residual model both fit on training years only)...")

    val_model = clone(best_model)
    val_model.fit(X_train_resid, resid_train_log)

    validation_years = np.arange(2015, 2024)
    dw = data_water.values  # (17, 24): cols 0-14 train, 15-23 test
    y_test_by_state = dw[:, N_TRAIN_YEARS:N_TRAIN_YEARS + N_TEST_YEARS]

    trend_only = trend_predict(trend_intercepts, trend_slopes,
                                np.arange(N_TRAIN_YEARS, N_TRAIN_YEARS + N_TEST_YEARS))  # (17, 9)
    pred_resid_log = val_model.predict(X_val_resid).reshape(N_STATES, N_TEST_YEARS)
    preds = trend_only * np.exp(pred_resid_log)   # trend + weather-driven correction, level space
    y_test_m = y_test_by_state.ravel()

    metrics = {}
    for i, st in enumerate(STATE_NAMES):
        actual = dw[i, 15:24]
        pred = preds[i]

        state_mean_pred = np.full(9, dw[i, :15].mean())
        persistence = dw[i, 14:23]
        linear_trend = np.polyval(np.polyfit(np.arange(15), dw[i, :15], 1), np.arange(15, 24))

        metrics[st] = {
            'ML_R2': r2_score(actual, pred),
            'ML_RMSE': np.sqrt(mean_squared_error(actual, pred)),
            'ML_MAPE_%': mean_absolute_percentage_error(actual, pred) * 100,
            'LogTrendOnly_R2': r2_score(actual, trend_only[i]),
            'StateMean_R2': r2_score(actual, state_mean_pred),
            'Persistence_R2': r2_score(actual, persistence),
            'LinearTrend_R2': r2_score(actual, linear_trend),
            'min_pred': pred.min(),
        }

    metrics_df = pd.DataFrame(metrics).T
    print("\n--- Per-state validation vs naive baselines ---")
    print(metrics_df.round(2))

    n_beat = int((metrics_df['ML_R2'] > 0).sum())
    print(f"\nStates where the ML model beats the state-average baseline (R2>0): {n_beat}/17")
    print(f"Any negative predicted water values? {'YES' if metrics_df['min_pred'].min() < 0 else 'NO'}"
          f"  (min = {metrics_df['min_pred'].min():.1f})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # --- per-state figure ---------------------------------------------------
    fig, axs = plt.subplots(5, 4, figsize=(12, 13), sharex=True)
    axs = axs.flatten()
    box = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5)
    for i, st in enumerate(STATE_NAMES):
        ax = axs[i]
        ax.plot(validation_years, y_test_by_state[i], color='black', marker='o',
                markersize=3, linewidth=1.2, label='Survey (actual)', zorder=5)
        ax.plot(validation_years, preds[i], color='#4285F4', marker='s', markersize=2,
                 linewidth=0.8, alpha=0.85, label='Trend + weather-residual prediction')
        m = metrics[st]
        ax.text(0.05, 0.95, f"$R^2$={m['ML_R2']:.2f}\nRMSE={m['ML_RMSE']:.0f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=6, bbox=box)
        ax.set_title(st, fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    for j in range(len(STATE_NAMES), len(axs)):
        axs[j].axis('off')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.035, 'Year', ha='center')
    fig.text(0.005, 0.5, 'Irrigation Water ($m^3$/1000 kg cotton lint)', va='center', rotation='vertical')
    fig.suptitle(f'Per-state validation - {best_model_name}', y=1.0, fontsize=11)
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.98])
    plt.savefig(FIGURES_DIR / 'figure_per_state_validation.tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)

    metrics_df.to_excel(TABLES_DIR / 'per_state_validation_metrics.xlsx')

    # --- pooled validation scatter ------------------------------------------
    yp = preds.flatten()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test_m, yp, s=18, alpha=0.6, color='#4285F4', edgecolor='none')
    lo = 0
    hi = max(y_test_m.max(), yp.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(f"Pooled validation (all 17 states) - {best_model_name}  $R^2$={r2_score(y_test_m, yp):.3f}", fontsize=10)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_scatter_plot.tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\nNote: pooled R2 above is high mainly because it includes BETWEEN-state")
    print("variance (state baselines differ by >10x). The per-state table is the")
    print("honest measure of year-to-year skill within a state.")

    return metrics_df


def project_future(best_model, data_water, features_train, features_val, important_indices,
                    list_X_predict_resid):
    print("\nRefitting trend + weather-residual model on all historical data (2000-2023) "
          "for future projection...")

    dw = data_water.values
    n_hist_years = N_TRAIN_YEARS + N_TEST_YEARS
    # Refit the trend using the FULL 24-year record now that 2015-2023 is no
    # longer held out -- a more reliable long-run rate than the 15-year fit
    # used for honest validation.
    trend_intercepts, trend_slopes = fit_state_log_trends(dw, n_hist_years)

    # BUG FIX: residual_targets_log(dw, ..., 0, n_hist_years) ravels dw[:, 0:24]
    # row-major, giving each state's 24 years CONTIGUOUS (state0's y2000..y2023,
    # then state1's, ...). But X_all_resid below is features_train (state-major
    # blocks of 15 years each) stacked on features_val (state-major blocks of 9
    # years each) -- i.e. ALL states' first 15 years, THEN all states' last 9
    # years. Those two row orders only agree for the first 15 rows; from row 15
    # on, X's state and y's state silently diverge (confirmed: X row 15 is
    # Arizona's 2000 weather, paired against Alabama's 2015 residual). Building
    # resid_all_log from the SAME two blocks (train years, then val years) as
    # X_all_resid, rather than one ravel across all 24 years, keeps them aligned.
    resid_train_refit = residual_targets_log(dw, trend_intercepts, trend_slopes, 0, N_TRAIN_YEARS)
    resid_val_refit = residual_targets_log(dw, trend_intercepts, trend_slopes, N_TRAIN_YEARS, N_TEST_YEARS)
    resid_all_log = np.concatenate([resid_train_refit, resid_val_refit])

    X_all_resid = np.vstack([
        np.hstack([features_train[:, important_indices], state_onehot(N_TRAIN_YEARS)]),
        np.hstack([features_val[:, important_indices], state_onehot(N_TEST_YEARS)]),
    ])
    final_model = clone(best_model)
    final_model.fit(X_all_resid, resid_all_log)

    future_years = np.arange(2024, 2024 + N_PREDICT_YEARS)
    trend_future = trend_predict(trend_intercepts, trend_slopes,
                                  np.arange(n_hist_years, n_hist_years + N_PREDICT_YEARS))  # (17, 26)

    reshaped_predictions = {}
    for label, Xp in zip(SSP_LABELS, list_X_predict_resid):
        resid_pred_log = final_model.predict(Xp).reshape(N_STATES, N_PREDICT_YEARS)
        p = trend_future * np.exp(resid_pred_log)
        reshaped_predictions[label] = p
        print(f"  {label}: min={p.min():.1f}, max={p.max():.1f}")

    all_future = np.concatenate([v.ravel() for v in reshaped_predictions.values()])
    print(f"\nAny negative projected values? {'YES' if all_future.min() < 0 else 'NO'} (min={all_future.min():.1f})")
    print("NOTE: the trend component is a 24-year log-linear fit extrapolated up to 26 years")
    print("further. For states with a strong historical trend (e.g. Kansas, ~-9%/year), that")
    print("compounds into very large changes by 2049 -- treat long-horizon values for those")
    print("states as illustrative of the *direction* implied by the trend, not precise forecasts.")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / 'future_water_predictions.xlsx'
    with pd.ExcelWriter(out_path) as writer:
        for label in SSP_LABELS:
            df = pd.DataFrame(reshaped_predictions[label], index=STATE_NAMES, columns=future_years)
            df.to_excel(writer, sheet_name=label)
    print(f"Future projections (2024-2049, per state, per SSP) saved to {out_path}")

    return reshaped_predictions


def main():
    setup_plot_style()

    data_water, data_abandonment, era5_features, weather_ssp = load_data()

    features_train, features_val, features_predict = build_features(era5_features, weather_ssp)

    # ------------------------------------------------------------------
    # Model = per-state log-linear trend (captures multi-decade structural
    # change, e.g. Ogallala-driven decline in Kansas, that year-to-year
    # weather can't) + a weather-driven correction fit on the DETRENDED
    # residual. Diagnostics showed the trend alone already beats the naive
    # state-average baseline on the true 2015-2023 hold-out in 8/17 states
    # (vs 0/17 for a flat weather-only model), and the weather-residual
    # correction improves several of those further (e.g. North Carolina
    # hold-out R2: -0.09 -> +0.27) while slightly hurting a couple others.
    # ------------------------------------------------------------------
    dw = data_water.values  # (17, 24)

    # Trend fit on TRAINING years only, so the 2015-2023 hold-out stays honest.
    trend_intercepts, trend_slopes = fit_state_log_trends(dw, N_TRAIN_YEARS)
    resid_train_log = residual_targets_log(dw, trend_intercepts, trend_slopes, 0, N_TRAIN_YEARS)

    # Feature/model selection targets the RESIDUAL, using training years only.
    important_indices = select_important_features(features_train, resid_train_log)

    state_ids_train = np.repeat(np.arange(N_STATES), N_TRAIN_YEARS)
    year_ids_train = np.tile(np.arange(N_TRAIN_YEARS), N_STATES)  # CV grouping key

    # State one-hot: every other model comparison in this project (CMIP-only
    # ensemble, the flexible ERA5 architecture) found state identity helps,
    # even after detrending -- some states' residual noise scale or small
    # systematic offset differs from others'. Not tried in this script before.
    X_train_resid = np.hstack([features_train[:, important_indices], state_onehot(N_TRAIN_YEARS)])
    X_val_resid = np.hstack([features_val[:, important_indices], state_onehot(N_TEST_YEARS)])

    print(f"X_train (residual model): {X_train_resid.shape}  "
          f"({len(important_indices)} weather features + {N_STATES} state one-hot)")

    # Residuals are centered on zero and can legitimately be negative, so
    # don't apply the irrigation-water nonnegativity filter here.
    best_model_name, best_model = select_model(X_train_resid, resid_train_log,
                                                state_ids_train, year_ids_train,
                                                enforce_nonnegative=False)

    validate_model(best_model_name, best_model, X_train_resid, resid_train_log, X_val_resid,
                    trend_intercepts, trend_slopes, data_water)

    list_X_predict_resid = [np.hstack([features_predict[l][:, important_indices], state_onehot(N_PREDICT_YEARS)])
                             for l in SSP_LABELS]
    project_future(best_model, data_water, features_train, features_val, important_indices,
                    list_X_predict_resid)

    print("\nDone. See the 'outputs' folder for figures and tables.")


if __name__ == "__main__":
    main()
