"""
Local execution of the JUST-FIXED 4_cotton_prediction_cmip_v2_2.ipynb
(reshape fix, MAPE*100 fix, non-negative meta-model, val/final leakage-split
fix, SSP4->SSP5 label fix), keeping its ORIGINAL 3xMLP ensemble architecture
(the diverse MLP+RF+KNN variant, tuning experiments, and state-identity
addition were explored separately and reverted -- back to this baseline per
explicit request, since it's the most faithful/minimal-change reproduction
of the notebook once the 5 confirmed bugs are fixed).

Colab-only paths (/content/drive/...) are swapped for the local data files
already used throughout this project; core logic mirrors the fixed notebook
cell-for-cell. Outputs go through the same generate_era5_10figs.py plotting
functions used for every other Fig 1-10 set in this project, for a
consistent, publication-quality result.
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
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "fixed_notebook_3mlp"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

sys.path.insert(0, str(SCRIPT_DIR))
import cotton_water_prediction as era5
import cotton_water_prediction_cmip_ensemble as cmip
import generate_era5_10figs as figs10
import run_original_notebook_cesm2 as run_cesm2  # reuse load_cmip_features (already-fixed reshape)

N_STATES = cmip.N_STATES
N_TRAIN_YEARS = cmip.N_TRAIN_YEARS
N_TEST_YEARS = cmip.N_TEST_YEARS
N_PREDICT_YEARS = cmip.N_PREDICT_YEARS
STATE_NAMES = cmip.STATE_NAMES
SSP_LABELS = cmip.SSP_LABELS
VALIDATION_SSP = "SSP1"


class PositiveKerasRegressor(RegressorMixin, BaseEstimator):
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
        return self.model.predict(X, verbose=0).flatten()


def make_mlp_pipeline(input_dim):
    return Pipeline([('scaler', StandardScaler()),
                      ('mlp', PositiveKerasRegressor(input_dim=input_dim, epochs=100, batch_size=32, verbose=0))])


def build_3mlp_stack(input_dim):
    base_models = [('mlp0', make_mlp_pipeline(input_dim)),
                   ('mlp1', make_mlp_pipeline(input_dim)),
                   ('mlp2', make_mlp_pipeline(input_dim))]
    meta_model = LinearRegression(fit_intercept=False, positive=True)
    return StackingRegressor(estimators=base_models, final_estimator=meta_model,
                              passthrough=False, cv=5, n_jobs=1)


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as _r2, mean_absolute_percentage_error as _mape, mean_squared_error as _mse


def make_4ssp_scatter_figure(y_true, preds_by_ssp, color, out_path):
    """2x2 grid, one panel per SSP -- matches the notebook's own
    'Visualization 3' layout (Fig 3-6): same true y_val, 4 different
    SSP-simulated weather realizations feeding predictions."""
    fig, axs = plt.subplots(2, 2, figsize=(5.2, 5.2), sharey=True, sharex=True)
    for i, label in enumerate(SSP_LABELS):
        y_pred = preds_by_ssp[label]
        r2 = _r2(y_true, y_pred)
        mape = _mape(y_true, y_pred) * 100
        rmse = np.sqrt(_mse(y_true, y_pred))
        ax = axs[i // 2, i % 2]
        ax.scatter(y_true, y_pred, marker='o', s=20, color=color, linewidth=0.5, alpha=0.5)
        ax.text(0.05, 0.94, f'$R^2$= {r2:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=7)
        ax.text(0.05, 0.86, f'MAPE= {mape:.2f}%', transform=ax.transAxes, ha='left', va='top', fontsize=7)
        ax.text(0.05, 0.78, f'RMSE= {rmse:.0f}', transform=ax.transAxes, ha='left', va='top', fontsize=7)
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle='--', color='#EA4335', lw=0.8)
        ax.set_aspect('equal', 'box')
        ax.set_title(label, fontsize=9)
    fig.text(0.5, 0.02, 'Survey Values ($m^3$)', ha='center')
    fig.text(0.01, 0.5, 'Predicted Values ($m^3$)', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.03, 0.03, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_4ssp_per_state_figure(model_name, y_val, preds_by_state_by_ssp, out_path, tables_dir):
    """Per-state time series with all 4 SSP-predicted lines overlaid;
    metrics pool all 4 SSPs x 9 years = 36 points/state -- matches the
    notebook's cell 10 (per-state validation, reviewer request) exactly."""
    validation_years = np.arange(2015, 2024)
    y_val_by_state = y_val.reshape(N_STATES, N_TEST_YEARS)
    ssp_colors = {'SSP1': '#9AA0A6', 'SSP2': '#EA4335', 'SSP3': '#FBBC04', 'SSP5': '#4285F4'}

    state_metrics = {}
    for i, st in enumerate(STATE_NAMES):
        pooled_actual = np.tile(y_val_by_state[i], len(SSP_LABELS))
        pooled_pred = np.concatenate([preds_by_state_by_ssp[label][i] for label in SSP_LABELS])
        state_metrics[st] = {
            'R2': _r2(pooled_actual, pooled_pred),
            'MAPE_%': _mape(pooled_actual, pooled_pred) * 100,
            'RMSE': np.sqrt(_mse(pooled_actual, pooled_pred)),
        }

    fig, axs = plt.subplots(5, 4, figsize=(12, 13), sharex=True)
    axs = axs.flatten()
    box = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5)
    for i, st in enumerate(STATE_NAMES):
        ax = axs[i]
        ax.plot(validation_years, y_val_by_state[i], color='black', marker='o', markersize=3,
                linewidth=1.2, label='Survey (actual)', zorder=5)
        for label in SSP_LABELS:
            ax.plot(validation_years, preds_by_state_by_ssp[label][i], color=ssp_colors[label],
                    marker='s', markersize=2, linewidth=0.8, alpha=0.8, label=label)
        m = state_metrics[st]
        ax.text(0.05, 0.95, f"MAPE={m['MAPE_%']:.1f}%\nRMSE={m['RMSE']:.0f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=6, bbox=box)
        ax.set_title(st, fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    for j in range(len(STATE_NAMES), len(axs)):
        axs[j].axis('off')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.035, 'Year', ha='center')
    fig.text(0.005, 0.5, 'Irrigation Water ($m^3$/1000 kg cotton lint)', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.02, 0.06, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(state_metrics).T.to_excel(tables_dir / 'per_state_validation_metrics_4ssp.xlsx')
    return state_metrics


def main():
    cmip.setup_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    data_water, data_abandonment, era5_features, weather_ssp = era5.load_data()
    features_train, features_val, features_predict = run_cesm2.load_cmip_features(weather_ssp)
    print(f"Training data shape: {features_train.shape}")
    print(f"Validation data shape: {features_val.shape} (CESM2 {VALIDATION_SSP}, simulated 2015-2023)")

    # Validation features for ALL 4 SSPs (not just VALIDATION_SSP) -- the
    # notebook's own Visualization 3 / per-state validation (cell 10)
    # evaluate against all 4 simulated-weather realizations of the same true
    # 2015-2023 target, in a 2x2 grid. VALIDATION_SSP=SSP1 remains special
    # only for X_complete (the final/future-only refit, matching cell 7).
    features_val_by_ssp = {label: era5.process_weather_data(weather_ssp[label], 420, end=9) for label in SSP_LABELS}

    dw = data_water.values
    y_train = dw[:, :N_TRAIN_YEARS].ravel()
    y_val = dw[:, N_TRAIN_YEARS:N_TRAIN_YEARS + N_TEST_YEARS].ravel()

    important_indices, importance_df = cmip.select_important_features(features_train, y_train)
    input_dim = len(important_indices)
    print(f"Selected {input_dim} weather features (no state identity)")

    X_train = features_train[:, important_indices]
    X_val = features_val[:, important_indices]
    X_val_by_ssp = {label: F[:, important_indices] for label, F in features_val_by_ssp.items()}

    print("\nTraining individual models (MLP / RandomForest / KNN, for the individual-model figures)...")
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

    print("\nModel validation results (CESM2, simulated 2015-2023, all 4 SSPs -- matches notebook cell 6):")
    val_metrics = {}
    for name, model in models.items():
        for label, X in X_val_by_ssp.items():
            y_pred = model.predict(X)
            r2 = r2_score(y_val, y_pred)
            mape = mean_absolute_percentage_error(y_val, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            val_metrics[f"{name}_{label}"] = {'R2': r2, 'MAPE_%': mape, 'RMSE': rmse}
            print(f"  {name} - {label}: R2={r2:.3f}, MAPE={mape:.2f}%, RMSE={rmse:.0f}")

    X_complete = np.vstack([X_train, X_val])
    y_complete = np.concatenate([y_train, y_val])
    print(f"\nComplete dataset for final refit: X={X_complete.shape}, y={y_complete.shape}")

    # Fix 2 -- leakage split: val_ensemble (X_train only, honest) vs
    # final_ensemble (X_complete, future-only). Original 3xMLP architecture.
    print("Fitting honest-holdout ensemble (val_ensemble, 3xMLP stack, X_train only, cv=5)...")
    val_ensemble = build_3mlp_stack(input_dim)
    val_ensemble.fit(X_train, y_train)
    val_meta = val_ensemble.final_estimator_
    print(f"val_ensemble meta-model weights: {dict(zip(['mlp0', 'mlp1', 'mlp2'], val_meta.coef_.round(3)))}")

    print("Refitting final ensemble for future projections (final_ensemble, X_complete, cv=5)...")
    final_ensemble = build_3mlp_stack(input_dim)
    final_ensemble.fit(X_complete, y_complete)
    final_meta = final_ensemble.final_estimator_
    print(f"final_ensemble meta-model weights: {dict(zip(['mlp0', 'mlp1', 'mlp2'], final_meta.coef_.round(3)))}")

    reshaped_predictions = {}
    for label in SSP_LABELS:
        Xp = features_predict[label][:, important_indices]
        pred = final_ensemble.predict(Xp)
        reshaped_predictions[label] = pred.reshape(N_STATES, N_PREDICT_YEARS)
        print(f"  {label}: min={pred.min():.1f}, max={pred.max():.1f}, negative predictions: {(pred < 0).sum()}")

    # --- Fig 1, 2 ---
    figs10.make_weather_trend_figure(features_train, features_val, features_predict, FIGURES_DIR / 'weather_figure.tiff')
    cmip.make_feature_importance_figure(importance_df, FIGURES_DIR / 'figure_Fimportance.tiff')

    # --- Fig 3: main (stacked) ensemble -- honest holdout (val_ensemble, Fix 2),
    # all 4 SSP-simulated weather realizations of the same true 2015-2023 target,
    # 2x2 grid -- matches the notebook's own Visualization 3 layout ---
    ensemble_pred_val_by_ssp = {label: val_ensemble.predict(X) for label, X in X_val_by_ssp.items()}
    make_4ssp_scatter_figure(y_val, ensemble_pred_val_by_ssp, '#A88EC0', FIGURES_DIR / 'figure_scatter_plot_ensemble.tiff')

    # SSP1 remains the headline number quoted in text (it's the scenario used
    # for X_complete/final_ensemble too), but all 4 are now in the figure/table.
    for label in SSP_LABELS:
        yp = ensemble_pred_val_by_ssp[label]
        r2 = r2_score(y_val, yp)
        mape = mean_absolute_percentage_error(y_val, yp) * 100
        rmse = np.sqrt(mean_squared_error(y_val, yp))
        print(f"Final stacked-ensemble validation (CESM2 {label} 2015-2023, honest holdout): "
              f"R2={r2:.3f}, MAPE={mape:.2f}%, RMSE={rmse:.0f}, "
              f"negative predictions: {(yp < 0).sum()}/{len(yp)}")

    # Residual calibration for Fig 9's uncertainty band still pools SSP1 only
    # (the scenario the final_ensemble future predictions are most directly
    # comparable to) to keep that mechanism unchanged from before.
    ensemble_pred_val = ensemble_pred_val_by_ssp[VALIDATION_SSP]
    log_resid_ratio = np.log(y_val / ensemble_pred_val)
    resid_log_mu, resid_log_sigma = log_resid_ratio.mean(), log_resid_ratio.std(ddof=1)
    print(f"Pooled validation residual-ratio log-normal ({VALIDATION_SSP}): mu={resid_log_mu:.3f}, "
          f"sigma={resid_log_sigma:.3f} (exp(sigma)={np.exp(resid_log_sigma):.2f}x)")

    # --- Fig 4-6: individual models, same 4-SSP 2x2 grid ---
    colors = {'MLP': '#EA4335', 'RandomForest': '#FBBC04', 'KNN': '#34A853'}
    indiv_preds_by_ssp = {}
    for name, model in models.items():
        preds_by_ssp = {label: model.predict(X) for label, X in X_val_by_ssp.items()}
        indiv_preds_by_ssp[name] = preds_by_ssp
        make_4ssp_scatter_figure(y_val, preds_by_ssp, colors[name], FIGURES_DIR / f'figure_scatter_plot_{name}.tiff')

    # --- Fig 7: per-state validation, pooling all 4 SSPs (36 pts/state) --
    # matches the notebook's cell 10 (per-state validation, reviewer request) ---
    preds_by_state_by_ssp = {label: ensemble_pred_val_by_ssp[label].reshape(N_STATES, N_TEST_YEARS)
                              for label in SSP_LABELS}
    make_4ssp_per_state_figure('Stacked 3xMLP ensemble (fixed notebook)', y_val, preds_by_state_by_ssp,
                                FIGURES_DIR / 'figure_scatter_plot_per_state.tiff', TABLES_DIR)

    # --- Fig 8-10: bubble chart / uncertainty band / box plot ---
    figs10.make_future_figures(reshaped_predictions, data_water, FIGURES_DIR, TABLES_DIR,
                                resid_log_mu=resid_log_mu, resid_log_sigma=resid_log_sigma)

    future_years = np.arange(2024, 2024 + N_PREDICT_YEARS)
    with pd.ExcelWriter(TABLES_DIR / 'future_water_predictions.xlsx') as writer:
        for label in SSP_LABELS:
            pd.DataFrame(reshaped_predictions[label], index=STATE_NAMES, columns=future_years).to_excel(writer, sheet_name=label)
    pd.DataFrame(val_metrics).T.to_excel(TABLES_DIR / 'individual_model_validation_metrics.xlsx')

    print(f"\nDone. 10 figures in {FIGURES_DIR}, tables in {TABLES_DIR}.")


if __name__ == "__main__":
    main()
