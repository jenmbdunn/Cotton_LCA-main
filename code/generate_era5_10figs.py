"""
Generates the full Fig 1-10 set (matching the original notebook's figure
inventory, same as cotton_water_prediction_cmip_ensemble.py produces) for the
CURRENT ERA5 + trend-decomposition model in cotton_water_prediction.py --
real ERA5 weather for 2000-2023, bias-corrected CMIP SSP only for 2024-2049,
state one-hot + MLP candidate added, the row-alignment bug fixed.

Differs from the CMIP-ensemble script's figures in one structural way: that
script branches the 2015-2023 "test" period into 4 different SSP-scenario
weather realizations (since even its "historical" years are free-running GCM
output, one run per SSP file) and plots a 2x2 grid per model. ERA5 has a
single real historical record, so Figs 1 and 3-6 here show ONE trajectory
for 2000-2023 (what actually happened) and branch into the 4 SSPs only for
the genuinely-future 2024-2049 period -- arguably the more honest version of
this figure, not just an adaptation.

Fig inventory (-> output filename):
  1  weather_figure.tiff              -- runoff trend, ERA5 history + SSP future
  2  figure_Fimportance.tiff          -- feature importance (3-panel)
  3  figure_scatter_plot_ensemble.tiff -- main model (trend+ExtraTrees residual)
  4  figure_scatter_plot_MLP.tiff      -- trend+MLP residual
  5  figure_scatter_plot_RandomForest.tiff -- trend+RandomForest residual
  6  figure_scatter_plot_Ridge.tiff    -- trend+Ridge residual (linear baseline)
  7  figure_scatter_plot_per_state.tiff -- 17-panel, MAPE+RMSE only
  8  figure_predictedbubble.tiff
  9  figure_uncertaintyplot_2d.tiff
  10 figure_boxplot.tiff

Outputs go to outputs_era5_10figs/ next to this script.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.base import clone

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_era5_10figs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

sys.path.insert(0, str(SCRIPT_DIR))
import cotton_water_prediction as era5
import cotton_water_prediction_cmip_ensemble as cmip

N_STATES = cmip.N_STATES
N_TRAIN_YEARS = cmip.N_TRAIN_YEARS
N_TEST_YEARS = cmip.N_TEST_YEARS
N_PREDICT_YEARS = cmip.N_PREDICT_YEARS
STATE_NAMES = cmip.STATE_NAMES
SSP_LABELS = cmip.SSP_LABELS
SSP_COLORS = cmip.SSP_COLORS


# ============================================================================
# Fig 1: weather trend -- real ERA5 history (2000-2023) + SSP future branches
# ============================================================================
def make_weather_trend_figure(features_train, features_val, features_predict, out_path):
    print("Creating weather trend visualization (Fig 1)...")

    def runoff_signal(F, n_years):
        return (F[:, 32:33].reshape(N_STATES, n_years).sum(axis=0) +
                F[:, 31:32].reshape(N_STATES, n_years).sum(axis=0) +
                F[:, 34:35].reshape(N_STATES, n_years).sum(axis=0))

    hist_line = np.concatenate([runoff_signal(features_train, N_TRAIN_YEARS),
                                 runoff_signal(features_val, N_TEST_YEARS)])

    fig, ax = plt.subplots(figsize=(3.6, 3))
    ax.plot(range(0, 24), hist_line, label='Historical (ERA5, observed)',
            linestyle='-', color='black', linewidth=1)

    x_future = range(24, 24 + N_PREDICT_YEARS)
    lines = {label: runoff_signal(features_predict[label], N_PREDICT_YEARS) for label in SSP_LABELS}
    avg_line = np.mean(np.vstack(list(lines.values())), axis=0)
    ax.plot(x_future, avg_line, label='Predicted Average', linestyle='--', color='black', linewidth=1)
    for label, color in zip(SSP_LABELS, ['red', 'green', 'purple', 'orange']):
        ax.plot(x_future, lines[label], label=label.lower(), linestyle='-', color=color, linewidth=0.4, alpha=0.6)

    years = np.arange(2000, 2050)
    ax.set_xticks(np.arange(0, 50, 5))
    ax.set_xticklabels(years[::5], rotation=90, ha='center')
    ax.set_ylabel('Runoff (m/year)')
    ax.legend(fontsize=6, frameon=False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Fig 3-6: validation scatter plots -- single panel (one real ERA5 trajectory,
# not a 2x2-per-SSP grid) for the main model + 3 individual comparison models
# ============================================================================
def make_scatter_figure(y_val, y_pred, color, out_path):
    r2 = r2_score(y_val, y_pred)
    mape = mean_absolute_percentage_error(y_val, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.scatter(y_val, y_pred, marker='o', s=20, color=color, linewidth=0.5, alpha=0.5)
    ax.text(0.05, 0.94, f'R2= {r2:.2f}', transform=ax.transAxes, ha='left', va='top')
    ax.text(0.05, 0.88, f'MAPE= {mape:.2f}%', transform=ax.transAxes, ha='left', va='top')
    ax.text(0.05, 0.82, f'RMSE= {rmse:.0f}', transform=ax.transAxes, ha='left', va='top')
    ax.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], linestyle='--', color='#EA4335', lw=0.8)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Survey Values (m3)')
    ax.set_ylabel('Predicted Values (m3)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Fig 7: per-state validation, MAPE+RMSE only (matches Fig 10 convention
# established in cotton_water_prediction_cmip_ensemble.py)
# ============================================================================
def make_per_state_figure(model_name, y_val, y_pred_by_state, out_path, tables_dir):
    validation_years = np.arange(2015, 2024)
    y_val_by_state = y_val.reshape(N_STATES, N_TEST_YEARS)

    state_metrics = {}
    for i, st in enumerate(STATE_NAMES):
        state_metrics[st] = {
            'R2': r2_score(y_val_by_state[i], y_pred_by_state[i]),
            'MAPE_%': mean_absolute_percentage_error(y_val_by_state[i], y_pred_by_state[i]) * 100,
            'RMSE': np.sqrt(mean_squared_error(y_val_by_state[i], y_pred_by_state[i])),
        }

    fig, axs = plt.subplots(5, 4, figsize=(12, 13), sharex=True)
    axs = axs.flatten()
    box = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5)
    for i, st in enumerate(STATE_NAMES):
        ax = axs[i]
        ax.plot(validation_years, y_val_by_state[i], color='black', marker='o', markersize=3,
                linewidth=1.2, label='Survey (actual)', zorder=5)
        ax.plot(validation_years, y_pred_by_state[i], color='#4285F4', marker='s', markersize=2,
                linewidth=0.8, alpha=0.85, label=f'Trend + {model_name} residual')
        m = state_metrics[st]
        ax.text(0.05, 0.95, f"MAPE={m['MAPE_%']:.1f}%\nRMSE={m['RMSE']:.0f}", transform=ax.transAxes,
                ha='left', va='top', fontsize=6, bbox=box)
        ax.set_title(st, fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    for j in range(len(STATE_NAMES), len(axs)):
        axs[j].axis('off')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.035, 'Year', ha='center')
    fig.text(0.005, 0.5, 'Irrigation Water (m3/1000 kg cotton lint)', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.02, 0.06, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    state_metrics_df = pd.DataFrame(state_metrics).T
    state_metrics_df.to_excel(tables_dir / 'per_state_validation_metrics.xlsx')
    print(state_metrics_df.round(2))


# ============================================================================
# Figs 8-10: bubble chart, uncertainty band, box plot -- adapted from
# cotton_water_prediction_cmip_ensemble.make_future_projection_figures to
# take a reshaped_predictions dict directly (trend x residual, already
# combined) instead of a single flexible model.
# ============================================================================
def make_future_figures(reshaped_predictions, data_water, out_dir, tables_dir,
                         resid_log_mu=0.0, resid_log_sigma=0.0, seed=0,
                         prod_band=(0.10, 0.90), combined_band=(0.25, 0.75)):
    """resid_log_mu/resid_log_sigma (optional): mean/std of a pooled
    log-normal fit to validation-period residual ratios (actual/predicted,
    log space), pooled across all states and years. When resid_log_sigma > 0,
    each of the 1000 production-share draws also gets an independent random
    per-(state, year) multiplicative perturbation sampled from this
    distribution, so the resulting band reflects demonstrated year-to-year
    prediction error, not just production-share weighting uncertainty.
    Default (0.0) reproduces the original behaviour exactly for callers that
    don't pass these. prod_band/combined_band: the (lo, hi) percentile range
    shown in Fig 9 (when resid_log_sigma > 0) for the production-share-only
    band and the +residual-error band respectively -- independently
    adjustable since the two bands represent different uncertainty sources."""
    future_years = np.arange(2024, 2024 + N_PREDICT_YEARS)

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

    print("Creating bubble chart (Fig 8)...")
    fig = plt.figure(figsize=(9.4, 6.2))
    divider = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 5),
                                left=0.11, right=0.76, bottom=0.09, top=0.88, wspace=0.12, hspace=0.04)
    main_ax = fig.add_subplot(divider[1, 0])
    for index, row in df_combined.iterrows():
        for col in df_combined.columns:
            value = row[col]
            color = '#A88EC0' if value > 0 else '#A4D5B1'
            main_ax.scatter(col, index, s=abs(value) / 20, color=color, alpha=0.5)
    colors = ['#9AA0A6', '#EA4335', '#FBBC04', '#4285F4']
    row_sum_ax = fig.add_subplot(divider[1, 1], sharey=main_ax)
    row_sum_ax.barh(df_combined.index, row_sums / N_PREDICT_YEARS, color=colors, alpha=0.5)
    row_sum_ax.set_xlabel('Horizontal Average \n($m^3$/1000kg cotton lint)', fontsize=6)
    col_sum_ax = fig.add_subplot(divider[0, 0], sharex=main_ax)
    bar_colors = ['#A88EC0' if v > 0 else '#A4D5B1' for v in col_sums]
    col_sum_ax.bar(df_combined.columns, col_sums / (N_STATES * 4), color=bar_colors, alpha=0.5)
    col_sum_ax.set_ylabel('Vertical Average\n($m^3$/1000kg cotton lint)', fontsize=6)
    col_sum_ax.tick_params(axis='x', labelsize=6)
    yticks = range(0, len(df_combined), 4)
    main_ax.set_yticks(yticks)
    main_ax.set_yticklabels(df_combined.index[yticks])
    main_ax.set_ylabel('States')
    main_ax.grid(True, linestyle='--', alpha=0.7)

    year_ticks = np.arange(future_years[0], future_years[-1] + 1, 5)  # 2024, 2029, 2034, ...
    main_ax.set_xticks(year_ticks)

    fig.savefig(out_dir / 'figure_predictedbubble.tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)

    if not cmip.SAMPLING_XLSX.exists():
        print(f"  {cmip.SAMPLING_XLSX.name} not found -- skipping uncertainty band and box plot figures.")
        return

    sampling_water_percent = pd.read_excel(cmip.SAMPLING_XLSX, usecols=lambda c: c not in ['Unnamed: 0']).values
    df_previous = pd.DataFrame(cmip.HISTORICAL_NATIONAL).set_index('Year')
    years = future_years
    all_average_values = [np.array(df_previous['Value'])]
    all_min_values, all_max_values = [], []

    rng = np.random.default_rng(seed)
    prod_lo, prod_hi = prod_band
    comb_lo, comb_hi = combined_band

    def national_totals(label, with_residual):
        """(n_draws, N_PREDICT_YEARS) national totals from production-share
        draws, optionally with an independent per-(state, year) residual-ratio
        draw layered on top (the model's demonstrated year-to-year prediction
        error, see module docstring caller)."""
        pred = reshaped_predictions[label]  # (N_STATES, N_PREDICT_YEARS)
        if with_residual and resid_log_sigma > 0:
            n_draws = sampling_water_percent.shape[0]
            perturb = np.exp(rng.normal(resid_log_mu, resid_log_sigma, size=(n_draws, *pred.shape)))
            perturbed = pred[None, :, :] * perturb
            return np.einsum('ns,nsy->ny', sampling_water_percent, perturbed)
        return np.dot(sampling_water_percent, pred)

    # Two uncertainty sources, kept separate so they can be shown as two bands:
    # production-share weighting alone (narrower, "known" uncertainty) vs. that
    # plus the model's demonstrated residual prediction error (wider, the
    # "how well can it predict year-to-year change" uncertainty).
    all_df_results, all_df_results_prod_only = {}, {}
    for label in SSP_LABELS:
        df_results = pd.DataFrame(national_totals(label, with_residual=True).T, index=years)
        df_prod_only = pd.DataFrame(national_totals(label, with_residual=False).T, index=years)
        all_df_results[label] = df_results
        all_df_results_prod_only[label] = df_prod_only
        all_average_values.append(df_results.mean(axis=1))
        if resid_log_sigma > 0:
            # Literal min/max here would be the same heavy-tailed, rare-draw-
            # dominated range fixed in Fig 9 below (see p_hi_max/p_lo_min) --
            # use the same combined_band percentiles instead, for consistency
            # and so Fig 10's boxes aren't dominated by the same tail draws.
            all_min_values.append(df_results.quantile(comb_lo, axis=1))
            all_max_values.append(df_results.quantile(comb_hi, axis=1))
        else:
            all_min_values.append(df_results.min(axis=1))
            all_max_values.append(df_results.max(axis=1))

    print(f"Creating uncertainty band plot (Fig 9), production-share band={prod_lo:.0%}-{prod_hi:.0%}, "
          f"+residual band={comb_lo:.0%}-{comb_hi:.0%}...")
    # Cap the shared y-axis to whichever band reaches further (+ padding), not
    # the full min-max -- with resid_log_sigma > 0 the min-max range is
    # dominated by rare extreme draws (heavy-tailed pooled log-normal) and would
    # otherwise force every panel's axis out to tens of thousands, hiding the
    # actual trend/bands.
    p_hi_max = max(
        max(df.quantile(comb_hi, axis=1).max() for df in all_df_results.values()),
        max(df.quantile(prod_hi, axis=1).max() for df in all_df_results_prod_only.values()),
    )
    p_lo_min = min(
        min(df.quantile(comb_lo, axis=1).min() for df in all_df_results.values()),
        min(df.quantile(prod_lo, axis=1).min() for df in all_df_results_prod_only.values()),
    )
    hist_min, hist_max = df_previous['Value'].min(), df_previous['Value'].max()
    ylo = min(p_lo_min, hist_min) * 0.9
    yhi = max(p_hi_max, hist_max) * 1.15

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4), sharex=True, sharey=True)
    axs = axs.flatten()
    for ax in axs:
        ax.plot(df_previous.index, df_previous['Value'], color='#9AA0A6', linestyle='--',
                linewidth=1.5, label='Historical', alpha=0.8)
    for i, label in enumerate(SSP_LABELS):
        df_results = all_df_results[label]
        ax = axs[i]
        if resid_log_sigma > 0:
            # Two uncertainty types, two alphas, two independently-chosen
            # percentile ranges (each band represents a different uncertainty
            # source, so its width need not match the other's): the lighter
            # band is production-share weighting + demonstrated residual
            # prediction error combined; the darker band is production-share
            # weighting alone.
            df_prod_only = all_df_results_prod_only[label]
            ax.fill_between(df_results.index, df_results.quantile(comb_lo, axis=1), df_results.quantile(comb_hi, axis=1),
                             color=SSP_COLORS[label], alpha=0.15,
                             label=f'{label}: + prediction residual error ({comb_lo:.0%}-{comb_hi:.0%} quantile)')
            ax.fill_between(df_prod_only.index, df_prod_only.quantile(prod_lo, axis=1), df_prod_only.quantile(prod_hi, axis=1),
                             color=SSP_COLORS[label], alpha=0.45,
                             label=f'{label}: production-share only ({prod_lo:.0%}-{prod_hi:.0%} quantile)')
            ax.set_ylim(ylo, yhi)
        else:
            # Original behaviour (no residual term): single full min-max band
            # plus mean line.
            ax.fill_between(df_results.index, df_results.min(axis=1), df_results.max(axis=1),
                             color=SSP_COLORS[label], alpha=0.3, label=f'{label} Range')
            ax.plot(df_results.index, df_results.mean(axis=1), color=SSP_COLORS[label],
                     linestyle='-', linewidth=2, label=f'{label} Average')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='upper left', frameon=False)
    fig.tight_layout()
    # if resid_log_sigma > 0:
        # Bands are empirical (Monte Carlo) quantiles, not a parametric CI --
        # state that plainly on the figure itself so it reads correctly even
        # detached from the caption/methods text.
        # fig.text(0.5, -0.02,
        #           'Shaded bands: empirical quantiles across 1000 Monte Carlo draws '
        #           '(production-share weighting x validation-calibrated residual error), '
        #           'not a parametric confidence interval.',
        #           ha='center', va='top', fontsize=6, color='#5f6368')
    fig.savefig(out_dir / 'figure_uncertaintyplot_2d.tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)

    if resid_log_sigma > 0:
        print(f"Creating box plot (Fig 10), min/max boxes = {comb_lo:.0%}/{comb_hi:.0%} quantile per year "
              f"(not literal min/max -- see make_future_figures docstring)...")
    else:
        print("Creating box plot (Fig 10)...")
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


def main():
    cmip.setup_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    data_water, data_abandonment, era5_features, weather_ssp = era5.load_data()
    features_train, features_val, features_predict = era5.build_features(era5_features, weather_ssp)
    dw = data_water.values

    trend_intercepts, trend_slopes = era5.fit_state_log_trends(dw, N_TRAIN_YEARS)
    resid_train_log = era5.residual_targets_log(dw, trend_intercepts, trend_slopes, 0, N_TRAIN_YEARS)
    resid_val_log = era5.residual_targets_log(dw, trend_intercepts, trend_slopes, N_TRAIN_YEARS, N_TEST_YEARS)

    important_indices, importance_df = cmip.select_important_features(features_train, resid_train_log)
    print(f"Selected {len(important_indices)} weather features")

    X_train_resid = np.hstack([features_train[:, important_indices], cmip.state_onehot(N_TRAIN_YEARS)])
    X_val_resid = np.hstack([features_val[:, important_indices], cmip.state_onehot(N_TEST_YEARS)])

    trend_val = era5.trend_predict(trend_intercepts, trend_slopes,
                                    np.arange(N_TRAIN_YEARS, N_TRAIN_YEARS + N_TEST_YEARS))
    y_val = dw[:, N_TRAIN_YEARS:N_TRAIN_YEARS + N_TEST_YEARS].ravel()

    # --- select the main model exactly as cotton_water_prediction.py does ---
    state_ids_train = np.repeat(np.arange(N_STATES), N_TRAIN_YEARS)
    year_ids_train = np.tile(np.arange(N_TRAIN_YEARS), N_STATES)
    best_model_name, best_model = era5.select_model(X_train_resid, resid_train_log,
                                                      state_ids_train, year_ids_train,
                                                      enforce_nonnegative=False)

    def fit_predict(model_builder):
        m = clone(model_builder)
        m.fit(X_train_resid, resid_train_log)
        resid_pred_log = m.predict(X_val_resid).reshape(N_STATES, N_TEST_YEARS)
        return trend_val * np.exp(resid_pred_log)

    main_pred_by_state = fit_predict(best_model)

    # --- Fig 1, 2 ---
    make_weather_trend_figure(features_train, features_val, features_predict, FIGURES_DIR / 'weather_figure.tiff')
    cmip.make_feature_importance_figure(importance_df, FIGURES_DIR / 'figure_Fimportance.tiff')

    # --- Fig 3: main model ---
    make_scatter_figure(y_val, main_pred_by_state.ravel(), '#A88EC0',
                         FIGURES_DIR / 'figure_scatter_plot_ensemble.tiff')

    # --- Fig 4-6: three individual comparison models (this script's candidate
    # pool has no KNN, unlike the CMIP-ensemble script -- MLP (explicitly
    # requested), RandomForest (closest relative of ExtraTrees) and Ridge (the
    # linear baseline) substitute for it here) ---
    individual_specs = [
        ('MLP', Pipeline([('scaler', StandardScaler()),
                           ('mlp', MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=2000,
                                                 early_stopping=True, n_iter_no_change=20, random_state=0))]),
         '#EA4335'),
        ('RandomForest', RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1),
         '#FBBC04'),
        ('Ridge', RidgeCV(alphas=np.logspace(-3, 4, 30)), '#34A853'),
    ]
    for name, builder, color in individual_specs:
        pred_by_state = fit_predict(builder)
        make_scatter_figure(y_val, pred_by_state.ravel(), color, FIGURES_DIR / f'figure_scatter_plot_{name}.tiff')

    # --- Fig 7 ---
    make_per_state_figure(best_model_name, y_val, main_pred_by_state, FIGURES_DIR / 'figure_scatter_plot_per_state.tiff', TABLES_DIR)

    # --- refit on all 24 years for future projection (same bug-fixed logic
    # as cotton_water_prediction.py.project_future) ---
    print(f"\nRefitting trend + {best_model_name} residual on all historical data for future projection...")
    n_hist_years = N_TRAIN_YEARS + N_TEST_YEARS
    trend_i_full, trend_s_full = era5.fit_state_log_trends(dw, n_hist_years)
    resid_train_refit = era5.residual_targets_log(dw, trend_i_full, trend_s_full, 0, N_TRAIN_YEARS)
    resid_val_refit = era5.residual_targets_log(dw, trend_i_full, trend_s_full, N_TRAIN_YEARS, N_TEST_YEARS)
    resid_all_log = np.concatenate([resid_train_refit, resid_val_refit])
    X_all_resid = np.vstack([X_train_resid, X_val_resid])
    final_model = clone(best_model)
    final_model.fit(X_all_resid, resid_all_log)

    trend_future = era5.trend_predict(trend_i_full, trend_s_full,
                                       np.arange(n_hist_years, n_hist_years + N_PREDICT_YEARS))
    reshaped_predictions = {}
    for label in SSP_LABELS:
        Xp = np.hstack([features_predict[label][:, important_indices], cmip.state_onehot(N_PREDICT_YEARS)])
        resid_pred_log = final_model.predict(Xp).reshape(N_STATES, N_PREDICT_YEARS)
        reshaped_predictions[label] = trend_future * np.exp(resid_pred_log)

    # --- Fig 8, 9, 10 ---
    make_future_figures(reshaped_predictions, data_water, FIGURES_DIR, TABLES_DIR)

    future_years = np.arange(2024, 2024 + N_PREDICT_YEARS)
    with pd.ExcelWriter(TABLES_DIR / 'future_water_predictions.xlsx') as writer:
        for label in SSP_LABELS:
            pd.DataFrame(reshaped_predictions[label], index=STATE_NAMES, columns=future_years).to_excel(writer, sheet_name=label)

    print(f"\nDone. 10 figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
