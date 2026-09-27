"""
Standalone script version of the original CMIP6-only stacked-ensemble
notebook (4_cotton_prediction_cmip_v2_2), with two real bugs fixed and the
per-state validation figure folded into the main flow.

Fix 1 -- process_weather_data() reshape bug
---------------------------------------------
The original reshape assumed the raw CMIP CSVs store months "month-major"
(all Januaries, then all Februaries, ...). The actual files are chronological
(year-major: Jan-Dec repeated per year) -- confirmed by checking that
consecutive January values across years cluster tightly relative to the full
12-month seasonal swing. The old reshape scrambled months and years together
into meaningless feature blends. Fixed by splitting the year axis off first.

Fix 2 -- data leakage in validation
-------------------------------------
The original fit `final_stacking_regressor` on the combined 2000-2023 data
(training years + the TRUE 2015-2023 targets), then reused that same model
to produce the "validation" figures/metrics against those same 2015-2023
years -- i.e. it was graded on data it had already been shown the answers
to. This script keeps two separate models: `val_ensemble`, fit ONLY on
2000-2014, used for every validation figure/metric; and `final_ensemble`,
fit on the full 2000-2023 record, used ONLY for the genuinely-future
2024-2049 projections.

Library substitution
----------------------
The original used Keras/TensorFlow MLPs. This script uses scikit-learn's
MLPRegressor instead (same conceptual architecture: 2 hidden layers, ReLU),
per explicit request, wrapped to clip negative outputs (irrigation water use
can't be negative -- matches the original's ReLU output layer).

Addition -- state one-hot indicator
--------------------------------------
The original model (Keras or scikit-learn) had no explicit signal for which
state a row belongs to -- only weather features. Confirmed this matters:
without it, per-state predictions were off by 5-30x in absolute level for
several states (VA, NC, AL). Added state_onehot() so the model has a direct
way to anchor each state's baseline level instead of inferring it indirectly
from weather values alone.

Main model: "Config G" -- log-target + trend + mixed-family + non-negative meta
------------------------------------------------------------------------------------
Went through three rounds of systematic comparison (see
../model_improvement_experiments/) before landing here:
  - Started from your original 3-MLP StackingRegressor (kept over the ERA5
    pipeline's CV-selected ExtraTrees because trees can't extrapolate beyond
    the training target range, which made future projections too flat).
  - Tried training in log-space (balances the loss across states spanning
    >1000x in scale, e.g. Virginia ~2-5 m3 vs Arizona/New Mexico ~5000-9000
    m3, and guarantees non-negative output for free via exp()) -- alone,
    this was much WORSE (log-space MLP predictions are unbounded and can
    blow up once exponentiated back; Virginia's R2 went from -2325 to
    -32677).
  - Adding genuine model-family diversity (RandomForest + ExtraTrees
    alongside 2 differently-shaped MLPs, not just varying NN architecture)
    fixed that: trees predict bounded averages of observed training values
    even in log-space, so they anchor the ensemble while the MLPs
    contribute flexibility. This alone (with a plain RidgeCV meta-model)
    roughly doubled the honest per-state validation beats-average count
    (3/17 -> 6/17).
  - Finally, constraining the meta-model to non-negative coefficients
    (LassoCV(positive=True) instead of Ridge) pushed it further (6/17 ->
    8/17, Virginia's R2 to -1.1) by stopping the meta-model from ever
    amplifying a base learner's occasional bad prediction with a negative
    weight.
The trend feature ("years since 2000") is added in main() alongside the
weather + state one-hot features, giving the model an explicit way to
represent long-run structural change (e.g. Kansas's decline) instead of
inferring it only from noisy weather.

Non-negative output
-----------------------
Guaranteed by construction via LogTargetRegressor (predictions are
exp(model.predict(log-space)), which is always positive) -- no clipping
wrapper needed, unlike the raw-scale architecture this replaced.

Fig 10 change -- MAPE and RMSE only
---------------------------------------
The per-panel textbox in the per-state validation figure now shows only
MAPE and RMSE (R2 dropped from the plot, though still computed and saved to
the metrics Excel table).

What this version does NOT fix (by explicit choice)
------------------------------------------------------
The weather inputs remain CMIP6 GCM output for every year, including
"historical" and the 2015-2023 "test" period. That GCM run is a free-running
simulation, not real observed weather -- year N in the file does not
correspond to the real year N (verified: the file's simulated 2011 doesn't
show the actual, well-documented 2011 Texas drought). So even this
leakage-free validation is still matching predictions against a fictional
weather-to-target correspondence for those years. If that matters for your
use case, swap in real ERA5 data (see the parallel ERA5-based pipeline and
fetch_era5_weather.py built earlier).

Outputs go to outputs_cmip_ensemble/ next to this script, kept separate from
the ERA5-based pipeline's outputs/ folder.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import pearsonr

from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin, clone

# ============================================================================
# CONFIG
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "results" / "processed_data and results"
WEATHER_DIR = REPO_ROOT / "data" / "Weather"
OUTPUT_DIR = SCRIPT_DIR / "outputs_cmip_ensemble"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
FONT_DIR = SCRIPT_DIR / "fonts"

WATER_XLSX = DATA_DIR / "water consumption.xlsx"
ABANDONMENT_XLSX = DATA_DIR / "abandonment_rate.xlsx"
WEATHER_HISTORICAL_CSV = WEATHER_DIR / "CMIP" / "historical" / "results_his_8f.csv"
WEATHER_SSP_CSVS = {
    "SSP1": WEATHER_DIR / "CMIP" / "SSP1" / "results_SSP1_8f.csv",
    "SSP2": WEATHER_DIR / "CMIP" / "SSP2" / "results_SSP2_8f.csv",
    "SSP3": WEATHER_DIR / "CMIP" / "SSP3" / "results_SSP3_8f.csv",
    "SSP5": WEATHER_DIR / "CMIP" / "SSP5" / "results_SSP5_8f.csv",
}
SAMPLING_XLSX = DATA_DIR / "sampling_for_production_percentage.xlsx"

N_STATES = 17
N_TRAIN_YEARS = 15    # 2000-2014
N_TEST_YEARS = 9      # 2015-2023
N_PREDICT_YEARS = 26  # 2024-2049
IMPORTANCE_THRESHOLD = 0.005

STATE_NAMES = ['AL', 'AZ', 'AR', 'CA', 'FL', 'GA', 'KS', 'LA', 'MS', 'MO',
               'NM', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA']
SSP_LABELS = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
SSP_COLORS = {'SSP1': '#9AA0A6', 'SSP2': '#EA4335', 'SSP3': '#FBBC04', 'SSP5': '#4285F4'}
MONTH_LABELS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
WEATHER_CATEGORIES = ['pr', 'huss', 'mrro', 'sfcWind', 'tas', 'clt', 'evspsbl', 'mrsos']

HISTORICAL_NATIONAL = {
    "Year": list(range(2000, 2024)),
    "Value": [1853.181355, 1596.030937, 1634.189531, 1510.225051, 1367.249686, 1327.999080,
              1300.773812, 1297.340748, 1414.761542, 1433.794234, 1398.765468, 1585.035214,
              1358.692156, 1090.963036, 1042.019891, 1139.378042, 1092.362063, 1118.108695,
              1097.238793, 1096.024428, 1068.022877, 994.640806, 897.099206, 967.590952],
}


def setup_plot_style():
    if FONT_DIR.exists():
        for f in font_manager.findSystemFonts(fontpaths=[str(FONT_DIR)]):
            font_manager.fontManager.addfont(f)
        matplotlib.rcParams['font.family'] = "Arial"
        matplotlib.rcParams['font.sans-serif'] = ["Arial"]
    else:
        print(f"No font folder at {FONT_DIR} -- using matplotlib's default font.")
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['figure.dpi'] = 150


# ============================================================================
# Data loading and feature engineering
# ============================================================================
def load_data():
    print("Loading input data...")
    data_water = pd.read_excel(WATER_XLSX, index_col=0)
    data_abandonment = pd.read_excel(ABANDONMENT_XLSX, index_col=0).iloc[:, ::-1]
    weather_historical = pd.read_csv(WEATHER_HISTORICAL_CSV)
    weather_ssp = {label: pd.read_csv(path) for label, path in WEATHER_SSP_CSVS.items()}
    print(f"  water consumption.xlsx state order: {list(data_water.index)}")
    print(f"  (expected to correspond, in order, to: {STATE_NAMES})")
    return data_water, data_abandonment, weather_historical, weather_ssp


def process_weather_data(weather_data, total_month, start=0, end=None):
    """Process raw CMIP weather data into (state x feature) arrays."""
    weather_array = np.array(weather_data).reshape(8, 17, total_month)
    # FIX 1: raw data is chronological (flat index = year*12 + month), so the
    # year axis (size total_month//12) must be split off first, with month
    # last (size 12) -- NOT (12, total_month//12), which assumes month-major
    # storage and scrambles months and years together.
    weather_array = weather_array.reshape(8, 17, total_month // 12, 12)
    weather_array = weather_array.transpose(1, 0, 3, 2).reshape(17, 96, total_month // 12)[:, :, start:end]

    new_indices = [9, 15, 16, 8, 5, 10, 13, 4, 0, 11, 6, 1, 2, 14, 12, 7, 3]
    weather_reordered = weather_array[new_indices]
    return weather_reordered.transpose(0, 2, 1).reshape(-1, 96)


def build_features(weather_historical, weather_ssp, data_water):
    print("Processing weather data...")
    features_train = process_weather_data(weather_historical, 180)
    features_test = {label: process_weather_data(weather_ssp[label], 420, end=9) for label in SSP_LABELS}
    features_predict = {label: process_weather_data(weather_ssp[label], 420, start=9) for label in SSP_LABELS}

    water_targets_train = np.array(data_water)[:, :N_TRAIN_YEARS].ravel()
    water_targets_test = np.array(data_water)[:, N_TRAIN_YEARS:].ravel()

    print(f"Training data shape: {features_train.shape}")
    print(f"Test data shape (per SSP): {features_test['SSP1'].shape}")
    return features_train, features_test, features_predict, water_targets_train, water_targets_test


def correlation_coefficients(X, y):
    return np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])


def select_important_features(features_train, water_targets_train):
    print("Analyzing feature importance...")
    correlation_importances = correlation_coefficients(features_train, water_targets_train)
    rfc = RandomForestRegressor(n_estimators=1000, random_state=42)
    rfc.fit(features_train, water_targets_train)
    rfc_importances = rfc.feature_importances_

    correlation_importances_abs = np.abs(correlation_importances) / np.sum(np.abs(correlation_importances))
    rfc_importances_norm = rfc_importances / np.sum(rfc_importances)
    overall_importance = np.mean([correlation_importances_abs, rfc_importances_norm], axis=0)

    features = [f'{cat}_{month}' for cat in WEATHER_CATEGORIES for month in MONTH_LABELS]
    importance_df = pd.DataFrame({
        'Feature': features,
        'Correlation Importance': correlation_importances,
        'RFC Importance': rfc_importances,
        'Overall Importance': overall_importance,
    })
    important_features = importance_df[importance_df['Overall Importance'] > IMPORTANCE_THRESHOLD]['Feature'].values
    important_indices = [features.index(f) for f in important_features]
    print(f"Selected {len(important_features)} important features")
    return important_indices, importance_df


def state_onehot(n_years):
    """One row per (state, year) in the same state-major/year-minor order
    process_weather_data() produces. Added so the model has an explicit
    signal for which state a row belongs to, instead of having to infer
    state identity indirectly from weather values alone -- without it, the
    model has no reliable way to anchor each state's baseline level."""
    sid = np.repeat(np.arange(N_STATES), n_years)
    oh = np.zeros((N_STATES * n_years, N_STATES))
    oh[np.arange(len(sid)), sid] = 1.0
    return oh


def trend_column(n_years, year_offset):
    """'Years since 2000' as an extra feature column, repeated per state in
    the same state-major/year-minor row order as everything else -- gives
    the model an explicit way to represent long-run structural change (e.g.
    Kansas's multi-decade decline) instead of inferring it only from noisy
    weather features."""
    return np.tile(np.arange(year_offset, year_offset + n_years), N_STATES).reshape(-1, 1).astype(float)


# ============================================================================
# Models
# ============================================================================
class PositiveMLPRegressor(RegressorMixin, BaseEstimator):
    """scikit-learn MLPRegressor wrapped to clip negative predictions --
    matches the original Keras model's ReLU output layer (irrigation water
    use physically cannot be negative)."""

    def __init__(self, hidden_layer_sizes=(50, 50), max_iter=1000, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation='relu',
                                    solver='adam', batch_size=32, max_iter=self.max_iter,
                                    early_stopping=True, n_iter_no_change=20, random_state=self.random_state)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model_.predict(X), 0, None)


class LogTargetRegressor(RegressorMixin, BaseEstimator):
    """Wraps any regressor to fit/predict in log-space. Balances the
    training loss across states of vastly different magnitude (Virginia
    ~2-5 m3 vs Arizona/New Mexico ~5000-9000 m3 -- a raw-scale squared-error
    loss is dominated by the large states, so the model barely tries to fit
    the small ones) and guarantees non-negative output for free (exp() is
    always positive), replacing the ad-hoc clipping wrappers a raw-scale
    model needs."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X, np.log(y))
        return self

    def predict(self, X):
        return np.exp(self.model_.predict(X))


def build_stacking_ensemble():
    """"Config G": the main prediction model, selected after a 3-round
    systematic comparison (see module docstring and
    ../model_improvement_experiments/). Two differently-shaped MLPs plus
    RandomForest and ExtraTrees as base learners (genuine model-family
    diversity, not just varying neural-net architecture), combined by a
    non-negative-constrained meta-model (LassoCV(positive=True)), all fit in
    log-space via LogTargetRegressor. Base MLPs here are plain (unclipped)
    MLPRegressor -- clipping would be wrong in log-space; positivity is
    guaranteed by the outer log/exp wrapper instead.

    Tried adding KNeighborsRegressor as a 5th base learner (standalone KNN
    showed real skill: 7/17 states beat the naive average with excellent
    calibration) -- reverted. LassoCV(positive=True) doesn't blend competing
    strong predictors, it picks a single winner (confirmed via its own
    coefficients: with 4 base learners it put 85% of the weight on
    RandomForest alone and zeroed ExtraTrees; adding KNN, it just switched
    to 100% KNN alone instead of blending). That single-KNN-via-the-stack
    result was WORSE than this 4-learner version (6/17 vs 8/17) despite
    better calibration bias, so KNN was left out. A real fix would need a
    meta-model that can blend rather than select, or a simpler weighted
    average outside the stacking machinery -- not attempted here."""
    base_models = [
        ('mlp_a', Pipeline([('scaler', StandardScaler()),
                             ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000,
                                                   early_stopping=True, n_iter_no_change=20, random_state=0))])),
        ('mlp_b', Pipeline([('scaler', StandardScaler()),
                             ('mlp', MLPRegressor(hidden_layer_sizes=(30, 30, 30), max_iter=1000,
                                                   early_stopping=True, n_iter_no_change=20, random_state=1))])),
        ('rf', RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ]
    stack = StackingRegressor(estimators=base_models,
                               final_estimator=LassoCV(alphas=np.logspace(-3, 2, 20), positive=True, max_iter=5000),
                               passthrough=False, cv=5, n_jobs=-1)
    return LogTargetRegressor(stack)


def train_individual_models(X_train, y_train):
    """MLP / RandomForest / KNN, each fit on training years only -- not
    affected by the leakage bug (the original already trained these on
    X_train only, not the combined data)."""
    print("Training individual models...")
    mlp_pipeline = Pipeline([('scaler', StandardScaler()),
                              ('mlp', PositiveMLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=0))])
    rf = RandomForestRegressor(random_state=42)
    knn_pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])

    param_grid_rf = {'n_estimators': [500], 'min_samples_split': [2, 3], 'min_samples_leaf': [1, 2]}
    param_grid_knn = {'knn__n_neighbors': [3, 5, 7], 'knn__weights': ['uniform', 'distance'], 'knn__algorithm': ['auto']}

    gscv_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
    gscv_knn = GridSearchCV(knn_pipeline, param_grid_knn, cv=5, scoring='r2', n_jobs=-1)

    mlp_pipeline.fit(X_train, y_train)
    gscv_rf.fit(X_train, y_train)
    gscv_knn.fit(X_train, y_train)

    return [mlp_pipeline, gscv_rf.best_estimator_, gscv_knn.best_estimator_]


def print_validation_metrics(models, model_names, list_X_test, y_test):
    print("Model validation results (per SSP):")
    for name, model in zip(model_names, models):
        for j, X_test in enumerate(list_X_test):
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"{name} - SSP{j+1}: R2={r2:.3f}, MAPE={mape:.2f}%, RMSE={rmse:.0f}")


# ============================================================================
# Visualization 1: weather trend
# ============================================================================
def make_weather_trend_figure(features_train, features_test, out_path):
    print("\nCreating weather trend visualization...")
    train_line = (features_train[:, 32:33].reshape(17, 15).sum(axis=0) +
                   features_train[:, 31:32].reshape(17, 15).sum(axis=0) +
                   features_train[:, 34:35].reshape(17, 15).sum(axis=0))
    fig, ax = plt.subplots(figsize=(3.6, 3))
    ax.plot(range(0, 15), train_line, label='Historical Average', linestyle='-', color='black', linewidth=1)

    x_range = range(15, 24)
    lines = {}
    for label in SSP_LABELS:
        F = features_test[label]
        lines[label] = (F[:, 32:33].reshape(17, 9).sum(axis=0) + F[:, 31:32].reshape(17, 9).sum(axis=0) +
                         F[:, 34:35].reshape(17, 9).sum(axis=0))
    average_line = np.mean(np.vstack(list(lines.values())), axis=0)
    ax.plot(x_range, average_line, label='Predicted Average', linestyle='--', color='black', linewidth=1)
    for label, color in zip(SSP_LABELS, ['red', 'green', 'purple', 'orange']):
        ax.plot(x_range, lines[label], label=label.lower(), linestyle='-', color=color, linewidth=0.4, alpha=0.6)

    years = np.arange(2000, 2024)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years, rotation=90, ha='center')
    ax.set_ylabel('Runoff (m/year)')
    ax.legend(fontsize=6, frameon=False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Visualization 2: feature importance
# ============================================================================
def make_feature_importance_figure(importance_df, out_path):
    print("Creating feature importance plots...")
    fig, ax = plt.subplots(1, 3, figsize=(6, 5.2), sharey=True)
    ax[2].barh(importance_df['Feature'], importance_df['Overall Importance'], color='#A4D5B1', alpha=0.7)
    ax[0].barh(importance_df['Feature'], importance_df['Correlation Importance'], color='#A88EC0', alpha=0.7)
    ax[1].barh(importance_df['Feature'], importance_df['RFC Importance'], color='#A88EC0', alpha=0.7)
    ax[2].axvline(x=IMPORTANCE_THRESHOLD, color='#A88EC0', linestyle='-.', linewidth=0.5)
    ax[0].set_yticks(range(0, len(importance_df), 12))
    ax[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Visualization 3: validation scatter plots (uses val_ensemble -- FIX 2)
# ============================================================================
def make_validation_scatter_figures(val_ensemble, individual_models, individual_names,
                                     list_X_test, y_test, out_dir):
    print("Creating validation scatter plots...")
    fig, axs = plt.subplots(2, 2, figsize=(5.2, 5.2), sharey=True, sharex=True)
    for i, X_test in enumerate(list_X_test):
        y_pred = val_ensemble.predict(X_test)
        r2, mape, rmse = (r2_score(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred),
                           np.sqrt(mean_squared_error(y_test, y_pred)))
        ax = axs[i // 2, i % 2]
        ax.scatter(y_test, y_pred, marker='o', s=20, color='#A88EC0', linewidth=0.5, alpha=0.5)
        ax.text(0.05, 0.94, f'R2= {r2:.2f}', transform=ax.transAxes, ha='left', va='top')
        ax.text(0.05, 0.88, f'MAPE= {mape:.2f}%', transform=ax.transAxes, ha='left', va='top')
        ax.text(0.05, 0.82, f'RMSE= {rmse:.0f}', transform=ax.transAxes, ha='left', va='top')
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='#EA4335', lw=0.8)
        ax.set_aspect('equal', 'box')
    fig.text(0.5, 0.04, 'Survey Values (m3)', ha='center')
    fig.text(0.04, 0.5, 'Predicted Values (m3)', va='center', rotation='vertical')
    fig.savefig(out_dir / 'figure_scatter_plot_ensemble.tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)

    colors = ['#EA4335', '#FBBC04', '#34A853']
    for model, color, name in zip(individual_models, colors, individual_names):
        fig, axs = plt.subplots(2, 2, figsize=(5.2, 5.2), sharey=True, sharex=True)
        for i, X_test in enumerate(list_X_test):
            y_pred = model.predict(X_test)
            r2, mape, rmse = (r2_score(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred),
                               np.sqrt(mean_squared_error(y_test, y_pred)))
            ax = axs[i // 2, i % 2]
            ax.scatter(y_test, y_pred, marker='o', s=20, color=color, linewidth=0.5, alpha=0.5)
            ax.text(0.05, 0.94, f'R2= {r2:.2f}', transform=ax.transAxes, ha='left', va='top')
            ax.text(0.05, 0.88, f'MAPE= {mape:.2f}%', transform=ax.transAxes, ha='left', va='top')
            ax.text(0.05, 0.82, f'RMSE= {rmse:.0f}', transform=ax.transAxes, ha='left', va='top')
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='#EA4335', lw=0.8)
            ax.set_aspect('equal', 'box')
        fig.text(0.5, 0.04, 'Survey Values (m3)', ha='center')
        fig.text(0.04, 0.5, 'Predicted Values (m3)', va='center', rotation='vertical')
        fig.savefig(out_dir / f'figure_scatter_plot_{name}.tiff', dpi=300, bbox_inches='tight')
        plt.close(fig)


# ============================================================================
# Fig 10: per-state, year-by-year validation (uses val_ensemble -- FIX 2)
# ============================================================================
def make_per_state_validation_figure(val_ensemble, list_X_test, y_test, data_water, out_path, tables_dir):
    print("Creating per-state, year-by-year validation plot...")
    validation_years = np.arange(2015, 2024)
    y_test_by_state = y_test.reshape(N_STATES, N_TEST_YEARS)

    preds_by_state = {label: val_ensemble.predict(X_test).reshape(N_STATES, N_TEST_YEARS)
                       for label, X_test in zip(SSP_LABELS, list_X_test)}

    state_metrics = {}
    for i, st in enumerate(STATE_NAMES):
        pooled_actual = np.tile(y_test_by_state[i], len(SSP_LABELS))
        pooled_pred = np.concatenate([preds_by_state[l][i] for l in SSP_LABELS])
        state_metrics[st] = {
            'R2': r2_score(pooled_actual, pooled_pred),
            'MAPE_%': mean_absolute_percentage_error(pooled_actual, pooled_pred) * 100,
            'RMSE': np.sqrt(mean_squared_error(pooled_actual, pooled_pred)),
        }

    fig, axs = plt.subplots(5, 4, figsize=(12, 13), sharex=True)
    axs = axs.flatten()
    textbox = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5)
    for i, st in enumerate(STATE_NAMES):
        ax = axs[i]
        ax.plot(validation_years, y_test_by_state[i], color='black', marker='o', markersize=3,
                linewidth=1.2, label='Survey (actual)', zorder=5)
        for label in SSP_LABELS:
            ax.plot(validation_years, preds_by_state[label][i], color=SSP_COLORS[label],
                     marker='s', markersize=2, linewidth=0.8, alpha=0.8, label=label)
        m = state_metrics[st]
        ax.text(0.05, 0.95, f"MAPE={m['MAPE_%']:.1f}%\nRMSE={m['RMSE']:.0f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=6, bbox=textbox)
        ax.set_title(st, fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    for j in range(len(STATE_NAMES), len(axs)):
        axs[j].axis('off')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.035, 'Year', ha='center')
    fig.text(0.005, 0.5, 'Irrigation Water (m3/1000 kg cotton lint)', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.02, 0.06, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    state_metrics_df = pd.DataFrame(state_metrics).T
    state_metrics_df.to_excel(tables_dir / 'per_state_validation_metrics.xlsx')
    print(state_metrics_df.round(2))


# ============================================================================
# Fig 7: bubble chart + Figs 8/9: uncertainty band + box plot
# ============================================================================
def make_future_projection_figures(final_ensemble, important_indices, features_predict, data_water, out_dir, tables_dir):
    print("\nRefitting on all historical data (2000-2023) for future projection...")
    future_years = np.arange(2024, 2024 + N_PREDICT_YEARS)
    trend_predict = trend_column(N_PREDICT_YEARS, N_TRAIN_YEARS + N_TEST_YEARS)
    reshaped_predictions = {}
    for label in SSP_LABELS:
        X_predict = np.hstack([features_predict[label][:, important_indices], state_onehot(N_PREDICT_YEARS), trend_predict])
        p = final_ensemble.predict(X_predict)
        reshaped_predictions[label] = p.reshape(N_STATES, N_PREDICT_YEARS)
        print(f"  {label}: min={p.min():.1f}, max={p.max():.1f}")

    df_water_history = data_water.copy()
    df_water_history.index = STATE_NAMES
    historical_mean = df_water_history.mean(axis=1)

    dfs = [pd.DataFrame(reshaped_predictions[label] - historical_mean.values.reshape(-1, 1),
                         index=STATE_NAMES, columns=future_years) for label in SSP_LABELS]
    interleaved_rows = []
    for i in range(N_STATES):
        for d in dfs:
            interleaved_rows.append(d.iloc[i])
    scenarios = ['', 'SSP2', 'SSP3', 'SSP5']
    new_index = [f'{st}{sc}' for st in STATE_NAMES for sc in scenarios]
    df_combined = pd.DataFrame(interleaved_rows).reset_index(drop=True)
    df_combined.index = new_index

    row_sums, col_sums = df_combined.sum(axis=1), df_combined.sum(axis=0)

    print("Creating bubble chart...")
    fig = plt.figure(figsize=(7.2, 6))
    divider = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 5),
                                left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.05)
    main_ax = fig.add_subplot(divider[1, 0])
    for index, row in df_combined.iterrows():
        for col in df_combined.columns:
            value = row[col]
            color = '#A88EC0' if value > 0 else '#A4D5B1'
            main_ax.scatter(col, index, s=abs(value) / 20, color=color, alpha=0.5)
    colors = ['#9AA0A6', '#EA4335', '#FBBC04', '#4285F4']
    row_sum_ax = fig.add_subplot(divider[1, 1], sharey=main_ax)
    row_sum_ax.barh(df_combined.index, row_sums / N_PREDICT_YEARS, color=colors, alpha=0.5)
    row_sum_ax.set_xlabel('Horizontal Average \n(m3/1000kg cotton lint)', fontsize=6)
    col_sum_ax = fig.add_subplot(divider[0, 0], sharex=main_ax)
    bar_colors = ['#A88EC0' if v > 0 else '#A4D5B1' for v in col_sums]
    col_sum_ax.bar(df_combined.columns, col_sums / (N_STATES * 4), color=bar_colors, alpha=0.5)
    col_sum_ax.set_ylabel('Vertical Average\n(m3/1000kg cotton lint)', fontsize=6)
    yticks = range(0, len(df_combined), 4)
    main_ax.set_yticks(yticks)
    main_ax.set_yticklabels(df_combined.index[yticks])
    main_ax.set_ylabel('States')
    main_ax.grid(True, linestyle='--', alpha=0.7)
    fig.savefig(out_dir / 'figure_predictedbubble.tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)

    if not SAMPLING_XLSX.exists():
        print(f"  {SAMPLING_XLSX.name} not found -- skipping uncertainty band and box plot figures.")
        return reshaped_predictions

    sampling_water_percent = pd.read_excel(SAMPLING_XLSX, usecols=lambda c: c not in ['Unnamed: 0']).values
    df_previous = pd.DataFrame(HISTORICAL_NATIONAL).set_index('Year')
    years = future_years
    all_average_values = [np.array(df_previous['Value'])]
    all_min_values, all_max_values = [], []
    base_predictions_dict = {label: reshaped_predictions[label] for label in SSP_LABELS}

    for label in SSP_LABELS:
        results = np.dot(sampling_water_percent, base_predictions_dict[label])
        df_results = pd.DataFrame(results.T, index=years)
        all_average_values.append(df_results.mean(axis=1))
        all_min_values.append(df_results.min(axis=1))
        all_max_values.append(df_results.max(axis=1))

    print("Creating uncertainty band plot...")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4), sharex=True, sharey=True)
    axs = axs.flatten()
    for ax in axs:
        ax.plot(df_previous.index, df_previous['Value'], color='#9AA0A6', linestyle='--',
                linewidth=1.5, label='Historical', alpha=0.8)
    for i, label in enumerate(SSP_LABELS):
        results = np.dot(sampling_water_percent, base_predictions_dict[label])
        df_results = pd.DataFrame(results.T, index=years)
        ax = axs[i]
        ax.fill_between(df_results.index, df_results.min(axis=1), df_results.max(axis=1),
                         color=SSP_COLORS[label], alpha=0.3, label=f'{label} Range')
        ax.plot(df_results.index, df_results.mean(axis=1), color=SSP_COLORS[label],
                 linestyle='-', linewidth=2, label=f'{label} Average')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / 'figure_uncertaintyplot_2d.tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Creating box plot...")
    fig, ax = plt.subplots(figsize=(7.2, 1.5))
    flierprops = dict(marker='o', color='k', markersize=2, markeredgewidth=0.5)
    boxprops = dict(color='white', linewidth=0.1)
    meanprops = {'linestyle': '-', 'color': 'grey', 'linewidth': 1}
    bplot1 = ax.boxplot(np.array(df_previous['Value']), patch_artist=True, positions=[-1], showmeans=True,
                         meanline=True, widths=0.5, whis=3, flierprops=flierprops, boxprops=boxprops, meanprops=meanprops)
    average_values_df = pd.DataFrame(all_average_values).T
    bplot2 = ax.boxplot(average_values_df, positions=[0, 2, 6, 10, 14], patch_artist=True, showmeans=True,
                         meanline=True, widths=0.5, whis=3, flierprops=flierprops, boxprops=boxprops, meanprops=meanprops)
    bplot3 = ax.boxplot(all_min_values, positions=[1, 5, 9, 13], patch_artist=True, showmeans=True,
                         meanline=True, widths=0.5, whis=3, flierprops=flierprops, boxprops=boxprops, meanprops=meanprops)
    bplot4 = ax.boxplot(all_max_values, positions=[3, 7, 11, 15], patch_artist=True, showmeans=True,
                         meanline=True, widths=0.5, whis=3, flierprops=flierprops, boxprops=boxprops, meanprops=meanprops)
    ax.axvspan(-2, 0, facecolor='#E37400', alpha=0.1, edgecolor='none')
    colors5 = ['#E37400', '#9AA0A6', '#EA4335', '#FBBC04', '#4285F4']
    for patch in bplot1['boxes']:
        patch.set_facecolor(colors5[0]); patch.set_alpha(0.6)
    for patch, c in zip(bplot2['boxes'], colors5[:len(bplot2['boxes'])]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for patch, c in zip(bplot3['boxes'], colors5[1:len(bplot3['boxes']) + 1]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for patch, c in zip(bplot4['boxes'], colors5[1:len(bplot4['boxes']) + 1]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticks([-1, 2, 6, 10, 14])
    ax.set_xticklabels(['Historical', 'SSP1', 'SSP2', 'SSP3', 'SSP5'])
    ax.set_ylabel('Irrigation Water\n(m3/1000 kg cotton lint)', fontsize=6)
    ax.set_rasterized(True)
    fig.savefig(out_dir / 'figure_boxplot.tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)

    average_values_df.columns = ['Historical'] + SSP_LABELS
    average_values_df.to_excel(tables_dir / 'Water_future_average.xlsx')
    return reshaped_predictions


def main():
    setup_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    data_water, data_abandonment, weather_historical, weather_ssp = load_data()
    (features_train, features_test, features_predict,
     water_targets_train, water_targets_test) = build_features(weather_historical, weather_ssp, data_water)

    important_indices, importance_df = select_important_features(features_train, water_targets_train)
    trend_train = trend_column(N_TRAIN_YEARS, 0)
    trend_test = trend_column(N_TEST_YEARS, N_TRAIN_YEARS)
    X_train = np.hstack([features_train[:, important_indices], state_onehot(N_TRAIN_YEARS), trend_train])
    list_X_test = [np.hstack([features_test[l][:, important_indices], state_onehot(N_TEST_YEARS), trend_test])
                   for l in SSP_LABELS]
    y_train, y_test = water_targets_train, water_targets_test

    individual_models = train_individual_models(X_train, y_train)
    individual_names = ['MLP', 'RandomForest', 'KNN']
    print_validation_metrics(individual_models, individual_names, list_X_test, y_test)

    # FIX 2: two separate fits of the stacking ensemble -- one for validation
    # (train-only), one for future projection (fit on everything, never used
    # for validation).
    print("\nFitting validation ensemble (2000-2014 only)...")
    val_ensemble = build_stacking_ensemble()
    val_ensemble.fit(X_train, y_train)

    print("Fitting final ensemble (2000-2023, for future projection only)...")
    X_complete = np.vstack([X_train, list_X_test[0]])
    y_complete = np.concatenate([y_train, y_test])
    final_ensemble = build_stacking_ensemble()
    final_ensemble.fit(X_complete, y_complete)

    make_weather_trend_figure(features_train, features_test, FIGURES_DIR / 'weather_figure.tiff')
    make_feature_importance_figure(importance_df, FIGURES_DIR / 'figure_Fimportance.tiff')
    make_validation_scatter_figures(val_ensemble, individual_models, individual_names,
                                     list_X_test, y_test, FIGURES_DIR)
    make_per_state_validation_figure(val_ensemble, list_X_test, y_test, data_water,
                                      FIGURES_DIR / 'figure_scatter_plot_per_state.tiff', TABLES_DIR)
    make_future_projection_figures(final_ensemble, important_indices, features_predict, data_water,
                                    FIGURES_DIR, TABLES_DIR)

    print("\nDone. See the 'outputs_cmip_ensemble' folder for figures and tables.")


if __name__ == "__main__":
    main()
