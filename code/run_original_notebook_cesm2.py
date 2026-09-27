"""
Faithful reproduction of the original notebook
(4_cotton_prediction_cmip_v2_2 (1).ipynb, from ~/Downloads), run locally with
the ORIGINAL CMIP6/CESM2 weather data -- i.e. WITHOUT the ERA5 substitution
used in run_original_notebook_era5.py. This is "your previous method": train
(2000-2014) on CESM2's own "historical" experiment run, validate (2015-2023)
on CESM2's SSP1 scenario run (the notebook's own choice -- confirmed from its
cell 7: X_complete = vstack([train, test_ssp1])), future (2024-2049) on
CESM2's SSP1/SSP2/SSP3/SSP5 runs. No bias correction is applied: train,
validation and future all come from the same self-consistent CMIP6/CESM2
scale, unlike the ERA5 sibling script where bias correction reconciles an
ERA5-trained model with CMIP-scale future data.

Same real-bug fix carried over from run_original_notebook_era5.py (see that
script's docstring for the full reasoning): PositiveKerasRegressor's ReLU
output layer guarantees each individual MLP predicts >= 0, but the notebook's
final StackingRegressor combined the three MLPs with a plain LinearRegression
meta-model, which CAN and did go negative. Fixed by constraining the
meta-model to LinearRegression(fit_intercept=False, positive=True).

State identity is NOT added as an input feature, matching the original
notebook and the ERA5 sibling script's current configuration.

Ensemble diversity fix: the notebook's final StackingRegressor stacked THREE
MLPs (same architecture, different random initializations) -- not a genuine
model-family ensemble. Confirmed empirically (both this script and the ERA5
sibling) that a single tuned RandomForest or KNN individually beats that
all-MLP stack on every metric, and the meta-model often collapsed to using
just 1-2 of the 3 MLPs anyway (weights like {mlp0: 0, mlp1: 0, mlp2: 1.0}) --
stacking three near-identical weak learners doesn't help when the family
itself is the weak point. The final stack here uses one MLP + the
already-tuned RandomForest + the already-tuned KNN (their GridSearchCV-picked
hyperparameters carry over, since StackingRegressor clones each estimator's
params, not its fitted state) instead, matching the genuine-diversity fix
already validated in cotton_water_prediction_cmip_ensemble.py.

Fix 2 -- data leakage in validation (same bug already found and fixed in
cotton_water_prediction_cmip_ensemble.py, inherited here from the notebook's
own cell 7). The notebook fit ONE StackingRegressor on X_complete
(train+validation combined) and then reused that SAME model to report
"validation" R2/MAPE/RMSE against X_val -- i.e. it graded itself on data it
had already been shown the answers to. This was always true here, but stayed
nearly invisible with an all-MLP stack (R2~0.84-0.88) because a fixed-epoch
MLP doesn't memorize individual training rows sharply. Once RandomForest and
KNN joined the stack above -- both of which CAN memorize/interpolate their
own training points almost exactly -- the leakage alone jumped validation R2
to 0.993, a spurious result, not genuine model skill. Fixed the same way as
the CMIP-ensemble script: two separate fits, `val_ensemble` (fit ONLY on
X_train, used for every validation figure/metric, Fig 3/7/9's residual
calibration) and `final_ensemble` (fit on X_complete, used ONLY for the
genuinely-future 2024-2049 projections, where no ground truth exists to leak
against).

Purpose: a direct before/after comparison against run_original_notebook_era5.py,
isolating what changes when the SAME architecture and fixes are fed CMIP6/
CESM2 weather (what the notebook actually used) instead of real ERA5
observations.

Outputs go to outputs_original_notebook_cesm2/ next to this script.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_original_notebook_cesm2"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

sys.path.insert(0, str(SCRIPT_DIR))
import cotton_water_prediction as era5
import cotton_water_prediction_cmip_ensemble as cmip
import generate_era5_10figs as figs10

N_STATES = cmip.N_STATES
N_TRAIN_YEARS = cmip.N_TRAIN_YEARS
N_TEST_YEARS = cmip.N_TEST_YEARS
N_PREDICT_YEARS = cmip.N_PREDICT_YEARS
STATE_NAMES = cmip.STATE_NAMES
SSP_LABELS = cmip.SSP_LABELS
VALIDATION_SSP = "SSP1"  # the notebook's own (arbitrary/undocumented) choice


class PositiveKerasRegressor(RegressorMixin, BaseEstimator):
    """Identical to the original notebook's class -- ReLU output layer
    guarantees each individual MLP predicts >= 0."""

    def __init__(self, input_dim, epochs=100, batch_size=32, verbose=0):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None

    def _build_model(self):
        model = Sequential([
            Dense(50, input_dim=self.input_dim, activation='relu'),
            Dense(50, activation='relu'),
            Dense(1, activation='relu'),
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fit(self, X, y):
        self.model = self._build_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                        verbose=self.verbose, validation_split=0.1)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X, verbose=0).flatten()


def make_mlp_pipeline(input_dim):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', PositiveKerasRegressor(input_dim=input_dim, epochs=100, batch_size=32, verbose=0)),
    ])


def load_cmip_features(weather_ssp):
    """CMIP6/CESM2-native features -- train from the "historical" run,
    validate from VALIDATION_SSP's own simulated 2015-2023, future from every
    SSP's simulated 2024-2049. No ERA5, no bias correction: everything is on
    the same self-consistent CESM2 scale, exactly as the notebook used it."""
    weather_historical = pd.read_csv(era5.WEATHER_HISTORICAL_CSV)
    features_train = era5.process_weather_data(weather_historical, 180)              # 2000-2014
    features_val = era5.process_weather_data(weather_ssp[VALIDATION_SSP], 420, end=9)  # 2015-2023
    features_predict = {
        label: era5.process_weather_data(weather_ssp[label], 420, start=9)  # 2024-2049
        for label in SSP_LABELS
    }
    return features_train, features_val, features_predict


def main():
    cmip.setup_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    data_water, data_abandonment, era5_features, weather_ssp = era5.load_data()
    features_train, features_val, features_predict = load_cmip_features(weather_ssp)
    print(f"Training data shape: {features_train.shape}")
    print(f"Validation data shape: {features_val.shape} (CESM2 {VALIDATION_SSP}, simulated 2015-2023)")

    dw = data_water.values
    y_train = dw[:, :N_TRAIN_YEARS].ravel()
    y_val = dw[:, N_TRAIN_YEARS:N_TRAIN_YEARS + N_TEST_YEARS].ravel()

    important_indices, importance_df = cmip.select_important_features(features_train, y_train)
    input_dim = len(important_indices)
    print(f"Selected {input_dim} weather features (no state identity)")

    X_train = features_train[:, important_indices]
    X_val = features_val[:, important_indices]

    print("\nTraining individual models (MLP / RandomForest / KNN)...")
    mlp_pipeline = make_mlp_pipeline(input_dim)
    rf = RandomForestRegressor(random_state=42)
    knn_pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])

    param_grid_rf = {'n_estimators': [500], 'min_samples_split': [2, 3], 'min_samples_leaf': [1, 2]}
    param_grid_knn = {'knn__n_neighbors': [3, 5, 7], 'knn__weights': ['uniform', 'distance'], 'knn__algorithm': ['auto']}
    gscv_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
    gscv_knn = GridSearchCV(knn_pipeline, param_grid_knn, cv=5, scoring='r2', n_jobs=-1)

    print("  fitting MLP...")
    mlp_pipeline.fit(X_train, y_train)
    print("  fitting RandomForest (grid search)...")
    gscv_rf.fit(X_train, y_train)
    print("  fitting KNN (grid search)...")
    gscv_knn.fit(X_train, y_train)
    models = {'MLP': mlp_pipeline, 'RandomForest': gscv_rf.best_estimator_, 'KNN': gscv_knn.best_estimator_}

    print(f"\nModel validation results (CESM2 {VALIDATION_SSP} simulated 2015-2023):")
    val_metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mape = mean_absolute_percentage_error(y_val, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        val_metrics[name] = {'R2': r2, 'MAPE_%': mape, 'RMSE': rmse}
        print(f"  {name}: R2={r2:.3f}, MAPE={mape:.2f}%, RMSE={rmse:.0f}")

    # --- combine train + validation weather, for the FINAL (future-only) refit
    # (matches the notebook's cell 7 exactly: X_complete = vstack([train, test_ssp1])) ---
    X_complete = np.vstack([X_train, X_val])
    y_complete = np.concatenate([y_train, y_val])
    print(f"\nComplete dataset for final refit: X={X_complete.shape}, y={y_complete.shape}")

    def build_diverse_stack():
        # Diverse-family stack (see module docstring): one MLP + the already-tuned
        # RandomForest + the already-tuned KNN, instead of three same-family MLPs.
        # StackingRegressor clones each estimator's PARAMS (not its fitted state)
        # before fitting, so reusing the already-fit gscv_rf.best_estimator_/
        # gscv_knn.best_estimator_ here carries over their GridSearchCV-picked
        # hyperparameters for free, safely, in both stacks built below.
        diverse_estimators = [
            ('mlp', make_mlp_pipeline(input_dim)),
            ('rf', models['RandomForest']),
            ('knn', models['KNN']),
        ]
        # non-negative weights, no intercept, so a non-negative combination of
        # non-negative base predictions can't go negative -- guaranteed by
        # construction, not just in practice (see module docstring).
        meta_model = LinearRegression(fit_intercept=False, positive=True)
        return StackingRegressor(estimators=diverse_estimators, final_estimator=meta_model,
                                  passthrough=False, cv=5, n_jobs=1)

    # Fix 2 -- data leakage in validation (see module docstring). Two separate
    # fits: val_ensemble (X_train only) for every validation figure/metric,
    # final_ensemble (X_complete) for future projections only.
    print("Fitting honest-holdout ensemble (val_ensemble, MLP+RF+KNN stack, "
          "X_train only, cv=5 -- this takes a while)...")
    val_ensemble = build_diverse_stack()
    val_ensemble.fit(X_train, y_train)
    val_meta = val_ensemble.final_estimator_
    print(f"val_ensemble meta-model weights: "
          f"{dict(zip(['mlp', 'rf', 'knn'], val_meta.coef_.round(3)))}")

    print("Refitting final ensemble for future projections (final_ensemble, "
          "X_complete = train+validation, cv=5 -- this takes a while)...")
    final_ensemble = build_diverse_stack()
    final_ensemble.fit(X_complete, y_complete)
    final_meta = final_ensemble.final_estimator_
    print(f"final_ensemble meta-model weights: "
          f"{dict(zip(['mlp', 'rf', 'knn'], final_meta.coef_.round(3)))}")

    # --- future predictions (final_ensemble only -- no leakage risk, no
    # ground truth exists yet for 2024-2049) ---
    reshaped_predictions = {}
    for label in SSP_LABELS:
        Xp = features_predict[label][:, important_indices]
        pred = final_ensemble.predict(Xp)
        reshaped_predictions[label] = pred.reshape(N_STATES, N_PREDICT_YEARS)
        print(f"  {label}: min={pred.min():.1f}, max={pred.max():.1f}, negative predictions: {(pred < 0).sum()}")

    # --- Fig 1, 2 ---
    figs10.make_weather_trend_figure(features_train, features_val, features_predict, FIGURES_DIR / 'weather_figure.tiff')
    cmip.make_feature_importance_figure(importance_df, FIGURES_DIR / 'figure_Fimportance.tiff')

    # --- Fig 3: main (stacked) ensemble -- honest holdout (val_ensemble, Fix 2) ---
    ensemble_pred_val = val_ensemble.predict(X_val)
    figs10.make_scatter_figure(y_val, ensemble_pred_val, '#A88EC0', FIGURES_DIR / 'figure_scatter_plot_ensemble.tiff')
    ens_r2 = r2_score(y_val, ensemble_pred_val)
    ens_mape = mean_absolute_percentage_error(y_val, ensemble_pred_val) * 100
    ens_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred_val))
    print(f"\nFinal stacked-ensemble validation (CESM2 {VALIDATION_SSP} 2015-2023, honest holdout): "
          f"R2={ens_r2:.3f}, MAPE={ens_mape:.2f}%, RMSE={ens_rmse:.0f}")
    print(f"Negative predictions in validation: {(ensemble_pred_val < 0).sum()} / {len(ensemble_pred_val)}")

    # Same Fig 9 uncertainty widening as the ERA5 sibling script: pool the
    # validation-period residual ratios into one log-normal so the band
    # reflects demonstrated prediction error, not just production-share
    # weighting. Here "validation-period" is CESM2's simulated 2015-2023, not
    # real weather -- worth remembering when comparing the two scripts' bands.
    log_resid_ratio = np.log(y_val / ensemble_pred_val)
    resid_log_mu, resid_log_sigma = log_resid_ratio.mean(), log_resid_ratio.std(ddof=1)
    print(f"Pooled validation residual-ratio log-normal: mu={resid_log_mu:.3f}, "
          f"sigma={resid_log_sigma:.3f} (exp(sigma)={np.exp(resid_log_sigma):.2f}x typical multiplicative spread)")

    # --- Fig 4-6: individual models ---
    colors = {'MLP': '#EA4335', 'RandomForest': '#FBBC04', 'KNN': '#34A853'}
    for name, model in models.items():
        y_pred = model.predict(X_val)
        figs10.make_scatter_figure(y_val, y_pred, colors[name], FIGURES_DIR / f'figure_scatter_plot_{name}.tiff')

    # --- Fig 7: per-state validation (stacked ensemble) ---
    pred_by_state = ensemble_pred_val.reshape(N_STATES, N_TEST_YEARS)
    figs10.make_per_state_figure('Stacked MLP ensemble', y_val, pred_by_state,
                                  FIGURES_DIR / 'figure_scatter_plot_per_state.tiff', TABLES_DIR)

    # --- Fig 8-10: bubble chart / uncertainty band / box plot ---
    figs10.make_future_figures(reshaped_predictions, data_water, FIGURES_DIR, TABLES_DIR,
                                resid_log_mu=resid_log_mu, resid_log_sigma=resid_log_sigma)

    # --- save tables ---
    future_years = np.arange(2024, 2024 + N_PREDICT_YEARS)
    with pd.ExcelWriter(TABLES_DIR / 'future_water_predictions.xlsx') as writer:
        for label in SSP_LABELS:
            pd.DataFrame(reshaped_predictions[label], index=STATE_NAMES, columns=future_years).to_excel(writer, sheet_name=label)
    pd.DataFrame(val_metrics).T.to_excel(TABLES_DIR / 'individual_model_validation_metrics.xlsx')

    print(f"\nDone. 10 figures in {FIGURES_DIR}, tables in {TABLES_DIR}.")


if __name__ == "__main__":
    main()
