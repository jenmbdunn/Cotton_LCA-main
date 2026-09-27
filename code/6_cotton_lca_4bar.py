"""
Cotton LCA — Nature-style 4-bar stacked figure (S1, S2, S3-S4, S5-S6)
Panels: (a) GHG-100, (b) Fossil energy, (c) Water consumption
Functional unit: 1,000 kg cotton lint

Reproducible: data parsed once from "Cotton LCA 0925.xlsx" sheet "LCA Results"
and frozen below as (low, high) tuples per (panel, scenario, category).
Run:  python 6_cotton_lca_4bar.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------------------------------------------------ #
# Matplotlib rcParams (Nature double-column, editable PDF text)
# ------------------------------------------------------------------ #
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#555555",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ------------------------------------------------------------------ #
# Frozen LCA data — values per 1,000 kg cotton lint
# Each entry: (low, high). Scalar contributions stored as (v, v).
# Source: "Cotton LCA 0925.xlsx", sheet "LCA Results".
# Scenarios in the spreadsheet: S1..S10.
#   S1  – Current (2025)
#   S2  – Baseline (2049)
#   S3  – 25% diesel electrified, high conversion (2049)
#   S4  – 25% diesel electrified, low conversion (2049)
#   S5  – S3 + renewable electricity
#   S6  – S4 + renewable electricity
#   S7  – Extreme full electrification (no diesel), high conv., renewable
#   S8  – Extreme full electrification (no diesel), low conv., renewable
#   S9  – Baseline + on-site solar PV (2049)
#   S10 – Baseline + on-site solar PV (2025)
# The figure shows 4 condensed bars: S1, S2, S3/S4 merged, S5/S6 merged.
# S7..S10 are kept in the data dicts for downstream reuse (sensitivity, SI).
# ------------------------------------------------------------------ #
SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]

# Category ordering and display names (stacked from bottom to top)
# Stack order — bottom to top of bar.
# Order interleaves small and large categories so each segment stays
# visually distinguishable rather than clustering tiny slices together.
# Small ag inputs sit at the bottom, fertilizer mid-stack, energy items
# above, with the smallest items (Sulfur, Electricity ginning) on top.
GHG_CATS = [
    "Herbicide", "Insecticide", "P2O5", "Potassium", "CaCO3",
    "Nitrogen", "N2O emission",
    "Natural gas",
    "Electricity (field ops.)",
    "Diesel",
    "Sulfur", "Electricity (ginning)",
]
FOS_CATS = [
    "Herbicide", "Insecticide", "P2O5", "Potassium", "CaCO3",
    "Nitrogen", "Natural gas", "Electricity (field ops.)",
    "Diesel", "Sulfur", "Electricity (ginning)",
]
WAT_CATS = [
    "Herbicide", "Insecticide", "P2O5", "Potassium", "CaCO3",
    "Nitrogen", "Natural gas", "Electricity (field ops.)",
    "Diesel", "Sulfur", "Electricity (ginning)", "Irrigation water",
]

GHG = {  # kg CO2-eq / 1,000 kg cotton lint  (LCA Results sheet rows 5-17)
    "S1": {"Herbicide": (92.99, 92.99), "Insecticide": (14.35, 14.35),
           "P2O5": (52.09, 52.09), "Potassium": (18.94, 18.94),
           "CaCO3": (0.01, 0.01), "Nitrogen": (294.29, 294.29),
           "N2O emission": (54.6, 1010.1), "Natural gas": (8.22, 43.26),
           "Electricity (field ops.)": (256.39, 256.39), "Diesel": (342.25, 342.25),
           "Sulfur": (0.21, 0.21), "Electricity (ginning)": (62.88, 71.96)},
    "S2": {"Herbicide": (61.98, 61.98), "Insecticide": (11.92, 11.92),
           "P2O5": (45.66, 45.66), "Potassium": (12.46, 12.46),
           "CaCO3": (0.01, 0.01), "Nitrogen": (282.01, 282.01),
           "N2O emission": (54.6, 1010.1), "Natural gas": (8.21, 43.2),
           "Electricity (field ops.)": (69.28, 69.28), "Diesel": (339.88, 339.88),
           "Sulfur": (0.09, 0.09), "Electricity (ginning)": (16.99, 19.45)},
    "S3": {"Herbicide": (61.98, 61.98), "Insecticide": (11.92, 11.92),
           "P2O5": (45.66, 45.66), "Potassium": (12.46, 12.46),
           "CaCO3": (0.01, 0.01), "Nitrogen": (282.01, 282.01),
           "N2O emission": (54.6, 1010.1), "Natural gas": (0, 0),
           "Electricity (field ops.)": (101, 101), "Diesel": (254.91, 254.91),
           "Sulfur": (0.09, 0.09), "Electricity (ginning)": (20.82, 39.59)},
    "S4": {"Herbicide": (61.98, 61.98), "Insecticide": (11.92, 11.92),
           "P2O5": (45.66, 45.66), "Potassium": (12.46, 12.46),
           "CaCO3": (0.01, 0.01), "Nitrogen": (282.01, 282.01),
           "N2O emission": (54.6, 1010.1), "Natural gas": (0, 0),
           "Electricity (field ops.)": (81.97, 81.97), "Diesel": (254.91, 254.91),
           "Sulfur": (0.09, 0.09), "Electricity (ginning)": (20.82, 39.59)},
    "S5": {"Herbicide": (61.98, 61.98), "Insecticide": (11.92, 11.92),
           "P2O5": (45.66, 45.66), "Potassium": (12.46, 12.46),
           "CaCO3": (0.01, 0.01), "Nitrogen": (282.01, 282.01),
           "N2O emission": (54.6, 1010.1), "Natural gas": (0, 0),
           "Electricity (field ops.)": (37.86, 37.86), "Diesel": (254.91, 254.91),
           "Sulfur": (0.09, 0.09), "Electricity (ginning)": (7.8, 14.84)},
    "S6": {"Herbicide": (61.98, 61.98), "Insecticide": (11.92, 11.92),
           "P2O5": (45.66, 45.66), "Potassium": (12.46, 12.46),
           "CaCO3": (0.01, 0.01), "Nitrogen": (282.01, 282.01),
           "N2O emission": (54.6, 1010.1), "Natural gas": (0, 0),
           "Electricity (field ops.)": (30.72, 30.72), "Diesel": (254.91, 254.91),
           "Sulfur": (0.09, 0.09), "Electricity (ginning)": (7.8, 14.84)},
    # Extreme full-electrification cases (post-2049): no diesel, renewable grid.
    "S7": {"Herbicide": (61.98, 61.98), "Insecticide": (11.92, 11.92),
           "P2O5": (45.66, 45.66), "Potassium": (12.46, 12.46),
           "CaCO3": (0.01, 0.01), "Nitrogen": (282.01, 282.01),
           "N2O emission": (54.6, 1010.1), "Natural gas": (0, 0),
           "Electricity (field ops.)": (73.53, 73.53), "Diesel": (0, 0),
           "Sulfur": (0.09, 0.09), "Electricity (ginning)": (7.8, 14.84)},
    "S8": {"Herbicide": (61.98, 61.98), "Insecticide": (11.92, 11.92),
           "P2O5": (45.66, 45.66), "Potassium": (12.46, 12.46),
           "CaCO3": (0.01, 0.01), "Nitrogen": (282.01, 282.01),
           "N2O emission": (54.6, 1010.1), "Natural gas": (0, 0),
           "Electricity (field ops.)": (44.99, 44.99), "Diesel": (0, 0),
           "Sulfur": (0.09, 0.09), "Electricity (ginning)": (7.8, 14.84)},
    # Baseline + on-site solar PV variants.
    "S9": {"Herbicide": (61.98, 61.98), "Insecticide": (11.92, 11.92),
           "P2O5": (45.66, 45.66), "Potassium": (12.46, 12.46),
           "CaCO3": (0.01, 0.01), "Nitrogen": (282.01, 282.01),
           "N2O emission": (54.6, 1010.1), "Natural gas": (8.21, 43.2),
           "Electricity (field ops.)": (31.52, 31.52), "Diesel": (412.53, 412.53),
           "Sulfur": (0.09, 0.09), "Electricity (ginning)": (6.37, 7.29)},
    "S10": {"Herbicide": (92.99, 92.99), "Insecticide": (14.35, 14.35),
           "P2O5": (52.09, 52.09), "Potassium": (18.94, 18.94),
           "CaCO3": (0.01, 0.01), "Nitrogen": (294.29, 294.29),
           "N2O emission": (54.6, 1010.1), "Natural gas": (8.22, 43.26),
           "Electricity (field ops.)": (30.51, 30.51), "Diesel": (342.25, 342.25),
           "Sulfur": (0.21, 0.21), "Electricity (ginning)": (7.48, 8.56)},
}

FOS = {  # MJ / 1,000 kg cotton lint  (LCA Results sheet rows 21-31)
    "S1": {"Herbicide": (1353.3, 1353.3), "Insecticide": (199.5, 199.5),
           "P2O5": (743, 743), "Potassium": (266.1, 266.1),
           "CaCO3": (0.2, 0.2), "Nitrogen": (4606.2, 4606.2),
           "Natural gas": (129.2, 680.2), "Electricity (field ops.)": (3554.7, 3554.7),
           "Diesel": (4519.2, 4519.2), "Sulfur": (3, 3),
           "Electricity (ginning)": (871.8, 997.7)},
    "S2": {"Herbicide": (955.5, 955.5), "Insecticide": (168, 168),
           "P2O5": (681.1, 681.1), "Potassium": (182.8, 182.8),
           "CaCO3": (0.1, 0.1), "Nitrogen": (4519.3, 4519.3),
           "Natural gas": (129.2, 680.2), "Electricity (field ops.)": (1145.2, 1145.2),
           "Diesel": (4519.2, 4519.2), "Sulfur": (1.5, 1.5),
           "Electricity (ginning)": (280.9, 321.4)},
    "S3": {"Herbicide": (955.5, 955.5), "Insecticide": (168, 168),
           "P2O5": (681.1, 681.1), "Potassium": (182.8, 182.8),
           "CaCO3": (0.1, 0.1), "Nitrogen": (4519.3, 4519.3),
           "Natural gas": (0, 0), "Electricity (field ops.)": (1669.6, 1669.6),
           "Diesel": (3389.4, 3389.4), "Sulfur": (1.5, 1.5),
           "Electricity (ginning)": (344.1, 654.5)},
    "S4": {"Herbicide": (955.5, 955.5), "Insecticide": (168, 168),
           "P2O5": (681.1, 681.1), "Potassium": (182.8, 182.8),
           "CaCO3": (0.1, 0.1), "Nitrogen": (4519.3, 4519.3),
           "Natural gas": (0, 0), "Electricity (field ops.)": (1355, 1355),
           "Diesel": (3389.4, 3389.4), "Sulfur": (1.5, 1.5),
           "Electricity (ginning)": (344.1, 654.5)},
    "S5": {"Herbicide": (955.5, 955.5), "Insecticide": (168, 168),
           "P2O5": (681.1, 681.1), "Potassium": (182.8, 182.8),
           "CaCO3": (0.1, 0.1), "Nitrogen": (4519.3, 4519.3),
           "Natural gas": (0, 0), "Electricity (field ops.)": (447.3, 447.3),
           "Diesel": (3389.4, 3389.4), "Sulfur": (1.5, 1.5),
           "Electricity (ginning)": (92.2, 175.3)},
    "S6": {"Herbicide": (955.5, 955.5), "Insecticide": (168, 168),
           "P2O5": (681.1, 681.1), "Potassium": (182.8, 182.8),
           "CaCO3": (0.1, 0.1), "Nitrogen": (4519.3, 4519.3),
           "Natural gas": (0, 0), "Electricity (field ops.)": (363, 363),
           "Diesel": (3389.4, 3389.4), "Sulfur": (1.5, 1.5),
           "Electricity (ginning)": (92.2, 175.3)},
    # Extreme full-electrification cases (post-2049): no diesel, renewable grid.
    "S7": {"Herbicide": (955.5, 955.5), "Insecticide": (168, 168),
           "P2O5": (681.1, 681.1), "Potassium": (182.8, 182.8),
           "CaCO3": (0.1, 0.1), "Nitrogen": (4519.3, 4519.3),
           "Natural gas": (0, 0), "Electricity (field ops.)": (868.7, 868.7),
           "Diesel": (0, 0), "Sulfur": (1.5, 1.5),
           "Electricity (ginning)": (92.2, 175.3)},
    "S8": {"Herbicide": (955.5, 955.5), "Insecticide": (168, 168),
           "P2O5": (681.1, 681.1), "Potassium": (182.8, 182.8),
           "CaCO3": (0.1, 0.1), "Nitrogen": (4519.3, 4519.3),
           "Natural gas": (0, 0), "Electricity (field ops.)": (531.5, 531.5),
           "Diesel": (0, 0), "Sulfur": (1.5, 1.5),
           "Electricity (ginning)": (92.2, 175.3)},
    # Baseline + on-site solar PV variants.
    "S9": {"Herbicide": (955.5, 955.5), "Insecticide": (168, 168),
           "P2O5": (681.1, 681.1), "Potassium": (182.8, 182.8),
           "CaCO3": (0.1, 0.1), "Nitrogen": (4519.3, 4519.3),
           "Natural gas": (129.2, 680.2), "Electricity (field ops.)": (372.4, 372.4),
           "Diesel": (5485.2, 5485.2), "Sulfur": (1.5, 1.5),
           "Electricity (ginning)": (75.2, 86.1)},
    "S10": {"Herbicide": (1353.3, 1353.3), "Insecticide": (199.5, 199.5),
           "P2O5": (743, 743), "Potassium": (266.1, 266.1),
           "CaCO3": (0.2, 0.2), "Nitrogen": (4606.2, 4606.2),
           "Natural gas": (129.2, 680.2), "Electricity (field ops.)": (365.2, 365.2),
           "Diesel": (4519.2, 4519.2), "Sulfur": (3, 3),
           "Electricity (ginning)": (89.6, 102.5)},
}

WAT = {  # m^3 / 1,000 kg cotton lint  (LCA Results sheet rows 36-47)
    "S1": {"Herbicide": (0.2371, 0.2371), "Insecticide": (0.025, 0.025),
           "P2O5": (1.0203, 1.0203), "Potassium": (0.153, 0.153),
           "CaCO3": (0.0002, 0.0002), "Nitrogen": (1.3206, 1.3206),
           "Natural gas": (0.0014, 0.0074), "Electricity (field ops.)": (1.2475, 1.2475),
           "Diesel": (0.3143, 0.3143), "Sulfur": (0.0056, 0.0056),
           "Electricity (ginning)": (0.3059, 0.3501), "Irrigation water": (681.3, 1373.9)},
    "S2": {"Herbicide": (0.1535, 0.1535), "Insecticide": (0.0184, 0.0184),
           "P2O5": (1.0029, 1.0029), "Potassium": (0.1356, 0.1356),
           "CaCO3": (0.0002, 0.0002), "Nitrogen": (1.2883, 1.2883),
           "Natural gas": (0.0014, 0.0073), "Electricity (field ops.)": (0.7442, 0.7442),
           "Diesel": (0.3044, 0.3044), "Sulfur": (0.0053, 0.0053),
           "Electricity (ginning)": (0.1825, 0.2089), "Irrigation water": (677.6, 1780.6)},
    "S3": {"Herbicide": (0.1535, 0.1535), "Insecticide": (0.0184, 0.0184),
           "P2O5": (1.0029, 1.0029), "Potassium": (0.1356, 0.1356),
           "CaCO3": (0.0002, 0.0002), "Nitrogen": (1.2883, 1.2883),
           "Natural gas": (0, 0), "Electricity (field ops.)": (1.0849, 1.0849),
           "Diesel": (0.2283, 0.2283), "Sulfur": (0.0053, 0.0053),
           "Electricity (ginning)": (0.2236, 0.4253), "Irrigation water": (677.6, 1780.6)},
    "S4": {"Herbicide": (0.1535, 0.1535), "Insecticide": (0.0184, 0.0184),
           "P2O5": (1.0029, 1.0029), "Potassium": (0.1356, 0.1356),
           "CaCO3": (0.0002, 0.0002), "Nitrogen": (1.2883, 1.2883),
           "Natural gas": (0, 0), "Electricity (field ops.)": (0.8805, 0.8805),
           "Diesel": (0.2283, 0.2283), "Sulfur": (0.0053, 0.0053),
           "Electricity (ginning)": (0.2236, 0.4253), "Irrigation water": (677.6, 1780.6)},
    "S5": {"Herbicide": (0.1535, 0.1535), "Insecticide": (0.0184, 0.0184),
           "P2O5": (1.0029, 1.0029), "Potassium": (0.1356, 0.1356),
           "CaCO3": (0.0002, 0.0002), "Nitrogen": (1.2883, 1.2883),
           "Natural gas": (0, 0), "Electricity (field ops.)": (0.5937, 0.5937),
           "Diesel": (0.2283, 0.2283), "Sulfur": (0.0053, 0.0053),
           "Electricity (ginning)": (0.1224, 0.2327), "Irrigation water": (677.6, 1780.6)},
    "S6": {"Herbicide": (0.1535, 0.1535), "Insecticide": (0.0184, 0.0184),
           "P2O5": (1.0029, 1.0029), "Potassium": (0.1356, 0.1356),
           "CaCO3": (0.0002, 0.0002), "Nitrogen": (1.2883, 1.2883),
           "Natural gas": (0, 0), "Electricity (field ops.)": (0.4818, 0.4818),
           "Diesel": (0.2283, 0.2283), "Sulfur": (0.0053, 0.0053),
           "Electricity (ginning)": (0.1224, 0.2327), "Irrigation water": (677.6, 1780.6)},
    # Extreme full-electrification cases (post-2049): no diesel, renewable grid.
    "S7": {"Herbicide": (0.1535, 0.1535), "Insecticide": (0.0184, 0.0184),
           "P2O5": (1.0029, 1.0029), "Potassium": (0.1356, 0.1356),
           "CaCO3": (0.0002, 0.0002), "Nitrogen": (1.2883, 1.2883),
           "Natural gas": (0, 0), "Electricity (field ops.)": (1.153, 1.153),
           "Diesel": (0, 0), "Sulfur": (0.0053, 0.0053),
           "Electricity (ginning)": (0.1224, 0.2327), "Irrigation water": (677.6, 1780.6)},
    "S8": {"Herbicide": (0.1535, 0.1535), "Insecticide": (0.0184, 0.0184),
           "P2O5": (1.0029, 1.0029), "Potassium": (0.1356, 0.1356),
           "CaCO3": (0.0002, 0.0002), "Nitrogen": (1.2883, 1.2883),
           "Natural gas": (0, 0), "Electricity (field ops.)": (0.7055, 0.7055),
           "Diesel": (0, 0), "Sulfur": (0.0053, 0.0053),
           "Electricity (ginning)": (0.1224, 0.2327), "Irrigation water": (677.6, 1780.6)},
    # Baseline + on-site solar PV variants.
    "S9": {"Herbicide": (0.1535, 0.1535), "Insecticide": (0.0184, 0.0184),
           "P2O5": (1.0029, 1.0029), "Potassium": (0.1356, 0.1356),
           "CaCO3": (0.0002, 0.0002), "Nitrogen": (1.2883, 1.2883),
           "Natural gas": (0.0014, 0.0073), "Electricity (field ops.)": (0.4943, 0.4943),
           "Diesel": (0.3695, 0.3695), "Sulfur": (0.0053, 0.0053),
           "Electricity (ginning)": (0.0999, 0.1143), "Irrigation water": (677.6, 1780.6)},
    "S10": {"Herbicide": (0.2371, 0.2371), "Insecticide": (0.025, 0.025),
           "P2O5": (1.0203, 1.0203), "Potassium": (0.153, 0.153),
           "CaCO3": (0.0002, 0.0002), "Nitrogen": (1.3206, 1.3206),
           "Natural gas": (0.0014, 0.0074), "Electricity (field ops.)": (0.4194, 0.4194),
           "Diesel": (0.3143, 0.3143), "Sulfur": (0.0056, 0.0056),
           "Electricity (ginning)": (0.1029, 0.1177), "Irrigation water": (681.3, 1373.9)},
}

# ------------------------------------------------------------------ #
# Color palette — consistent per category across the three panels
# ------------------------------------------------------------------ #
PALETTE = {
    "Herbicide":                "#9ED4E8",
    "Insecticide":              "#F0D34F",
    "P2O5":                     "#B5D693",
    "Potassium":                "#F0B89B",
    "CaCO3":                    "#DCDCDC",
    "Nitrogen":                 "#6C5B9C",
    "N2O emission":             "#3E2E72",
    "Natural gas":              "#6B3E1E",
    "Utility/Industrial Boiler": "#8B5A2B",
    "Electricity (field ops.)": "#19A6B5",
    "Diesel":                   "#E16E1E",
    "Farming Tractor":          "#B85C00",
    "Sulfur":                   "#8C7B5A",
    "Electricity (ginning)":    "#0F6F7A",
    "Irrigation water":         "#2E73AB",
}
DISPLAY_NAME = {
    "Herbicide": "Herbicide",
    "Insecticide": "Insecticide",
    "P2O5": "P$_2$O$_5$",
    "Potassium": "K$_2$O",
    "CaCO3": "CaCO$_3$",
    "Nitrogen": "N fertilizer",
    "N2O emission": "N$_2$O emission",
    "Natural gas": "Natural gas (well-to-wheel)",
    "Utility/Industrial Boiler": "Utility/industrial boiler (NG combustion)",
    "Electricity (field ops.)": "Electricity (field ops.)",
    "Diesel": "Diesel (well-to-wheel)",
    "Farming Tractor": "Farming tractor (diesel combustion)",
    "Sulfur": "Sulfur",
    "Electricity (ginning)": "Electricity (ginning)",
    "Irrigation water": "Irrigation water",
}

# ------------------------------------------------------------------ #
# Reduce each panel to 5 bars
#   bar 1 = S1          (mid, low, high per category)
#   bar 2 = S2          (mid, low, high per category)
#   bar 3 = S3-S4 merged: low = min(S3.low, S4.low), high = max(S3.high, S4.high)
#   bar 4 = S5-S6 merged: low = min(S5.low, S6.low), high = max(S5.high, S6.high)
#   bar 5 = S10         2025 baseline + on-site solar PV
# Total uncertainty per bar = linear sum (correlated worst-case) of category ranges.
# ------------------------------------------------------------------ #

def merge(d_a, d_b, cats):
    """Element-wise envelope of two scenarios across categories."""
    out = {}
    for c in cats:
        la, ha = d_a.get(c, (0, 0))
        lb, hb = d_b.get(c, (0, 0))
        out[c] = (min(la, lb), max(ha, hb))
    return out

def take(d, cats):
    return {c: d.get(c, (0, 0)) for c in cats}

def build_bars(panel_data, cats):
    # Display labels remain S1..S5 along the x-axis to preserve the existing
    # figure ordering; the 5th bar carries the S10 contribution data.
    bars = [
        ("S1",     take(panel_data["S1"], cats)),
        ("S2",     take(panel_data["S10"], cats)),
        ("S3",     take(panel_data["S2"], cats)),
        ("S4",  merge(panel_data["S3"], panel_data["S4"], cats)),
    ]
    return bars

# ------------------------------------------------------------------ #
# Plotting helpers
# ------------------------------------------------------------------ #

def luminance(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def draw_panel(ax, panel_label, panel_title, units_label, bars, cats,
               value_fmt="{:.0f}", min_label_frac=0.05,
               draw_sub_errorbars=True):
    n_bars = len(bars)
    x = np.arange(n_bars)
    bar_w = 0.32   # narrower bars leave horizontal room for annotations

    # Per-bar quantities
    mids = np.zeros(n_bars)              # midpoint total
    err_low = np.zeros(n_bars)           # |mid - low_total|
    err_high = np.zeros(n_bars)          # |high_total - mid|
    high_totals = np.zeros(n_bars)

    # Draw stacks (midpoint values), keep track of stack positions per bar
    for i, (label, data) in enumerate(bars):
        bottom = 0.0
        low_total = 0.0
        high_total = 0.0
        for c in cats:
            lo, hi = data.get(c, (0, 0))
            mid = 0.5 * (lo + hi)
            low_total += lo
            high_total += hi
            if mid <= 0:
                continue
            ax.bar(x[i], mid, width=bar_w, bottom=bottom,
                   color=PALETTE[c], edgecolor="white", linewidth=0.4,
                   zorder=2)
            bottom += mid
        mids[i] = 0.5 * (low_total + high_total)
        err_low[i] = mids[i] - low_total
        err_high[i] = high_total - mids[i]
        high_totals[i] = high_total

    # Total error bars on midpoint totals (larger, semi-transparent so
    # stacked segment labels behind the line remain readable)
    ax.errorbar(x, mids, yerr=[err_low, err_high], fmt="none",
                ecolor="#1a1a1a", elinewidth=1.0, capsize=4, capthick=1.0,
                alpha=0.55, zorder=6)

    # Precompute headroom-based offsets now so we can place the annotations.
    y_top_max = max(high_totals.max(), mids.max())
    headroom = 0.08 * y_top_max

    # Bar-total annotations, styled as an engineering dimension line
    # (ISO/ASME linear-dimension convention): a vertical double-headed
    # arrow beside the bar spans 0 to the bar's total height, closed off
    # by short horizontal extension (witness) lines at top and bottom,
    # with the total value rotated 90 deg and set to the left of the
    # dimension line — visually distinct from the italic in-segment
    # sub-bar value labels drawn inside the stack.
    dim_offset = bar_w / 2 + 0.14   # dimension line offset from bar center
    ext_overshoot = 0.05            # extension line overshoot past the dimension line
    ext_gap = 0.03                  # gap between bar edge and extension line start
    text_gap = 0.07                 # gap between dimension line and text
    for i in range(n_bars):
        dim_x = x[i] + dim_offset
        top_y = mids[i]
        ax.plot([x[i] + bar_w / 2 + ext_gap, dim_x + ext_overshoot], [0, 0],
                color="#555555", linewidth=0.5, zorder=7)
        ax.plot([x[i] + bar_w / 2 + ext_gap, dim_x + ext_overshoot], [top_y, top_y],
                color="#555555", linewidth=0.5, zorder=7)
        ax.annotate("", xy=(dim_x, top_y), xytext=(dim_x, 0),
                    arrowprops=dict(arrowstyle="<->", color="#555555",
                                     linewidth=0.6, shrinkA=0, shrinkB=0),
                    zorder=7)
        ax.text(dim_x - text_gap, top_y / 2, value_fmt.format(top_y),
                ha="center", va="center", rotation=90,
                fontsize=7, fontweight="bold", fontstyle="normal",
                color="#1a1a1a", zorder=8)

    # Sub-error bars on individual stacked segments with nonzero uncertainty
    # Drawn at the cumulative top of each segment (using midpoint stacking).
    if draw_sub_errorbars:
        min_seg_frac_for_eb = 0.012  # skip vanishingly thin segments
        for i, (label, data) in enumerate(bars):
            bottom = 0.0
            for c in cats:
                lo, hi = data.get(c, (0, 0))
                mid = 0.5 * (lo + hi)
                if mid <= 0:
                    bottom += mid
                    continue
                half_range = 0.5 * (hi - lo)
                seg_frac = mid / mids[i] if mids[i] > 0 else 0
                if half_range > 0 and seg_frac >= min_seg_frac_for_eb:
                    seg_top = bottom + mid
                    ax.errorbar(x[i], seg_top, yerr=half_range, fmt="none",
                                ecolor="#1a1a1a", elinewidth=0.7,
                                capsize=2.5, capthick=0.7,
                                alpha=0.55, zorder=5)
                bottom += mid

    # In-segment value labels (skip small slices)
    for i, (label, data) in enumerate(bars):
        bottom = 0.0
        for c in cats:
            lo, hi = data.get(c, (0, 0))
            mid = 0.5 * (lo + hi)
            if mid > 0:
                frac = mid / mids[i] if mids[i] > 0 else 0
                if frac >= min_label_frac:
                    txt_color = "white" if luminance(PALETTE[c]) < 0.55 else "#222222"
                    ax.text(x[i], bottom + mid / 2, value_fmt.format(mid),
                            ha="center", va="center",
                            fontsize=5.6, fontstyle="italic",
                            color=txt_color, zorder=5)
                bottom += mid

    # y_top_max and headroom were computed above for annotation placement.

    # Axis cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars])
    # Headroom must hold the upper error cap, the midpoint annotation, and
    # the (low – high) range line above it.
    ax.set_ylim(0, y_top_max + headroom * 1.2)
    # Slight padding on both sides; right side keeps modest annotation room.
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.70)
    ax.set_ylabel(units_label)
    ax.yaxis.grid(True, color="#999999", alpha=0.18, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", pad=2)

    # Panel label (bold lowercase, top-left, outside axes)
    ax.text(-0.13, 1.04, panel_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

# ------------------------------------------------------------------ #
# Figure
# ------------------------------------------------------------------ #

def make_figure(out_stem):
    # 183 mm wide x 175 mm tall — Nature double-column width, full cap.
    fig_w_in = 183 / 25.4
    fig_h_in = 175 / 25.4
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Stacked 3 rows x 1 col, panels extended to full figure width.
    # Panel (c) is shorter — water has fewer line items than GHG/FOS.
    # Tight hspace + larger bottom to give the legend more room.
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        left=0.14, right=0.97, top=0.975, bottom=0.17,
        hspace=0.22,
        height_ratios=[1.0, 1.0, 0.7],
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    draw_panel(ax1, "a",
               "GHG-100 (climate change)",
               "kg CO$_2$-eq per 1,000 kg lint",
               build_bars(GHG, GHG_CATS), GHG_CATS,
               value_fmt="{:.0f}",
               draw_sub_errorbars=False)
    draw_panel(ax2, "b",
               "Fossil resource use",
               "MJ per 1,000 kg lint",
               build_bars(FOS, FOS_CATS), FOS_CATS,
               value_fmt="{:.0f}",
               draw_sub_errorbars=False)
    draw_panel(ax3, "c",
               "Water consumption",
               "m$^3$ per 1,000 kg lint",
               build_bars(WAT, WAT_CATS), WAT_CATS,
               value_fmt="{:.0f}")

    # Sub-line under x-tick labels (scenario descriptions) — short enough to avoid overlap
    desc = {
        "S1": "Current Baseline",
        "S2": "Current Renewable Grid\nOn-Site Solar PV",
        "S3": "2049 Evolving Grid Baseline",
        "S4": "2049 Evolving Grid\n25% Electrification",
    }
    for ax in (ax1, ax2, ax3):
        new_labels = [f"{t.get_text()}\n{desc[t.get_text()]}" for t in ax.get_xticklabels()]
        ax.set_xticklabels(new_labels, fontsize=6.0, linespacing=1.15)
        ax.tick_params(axis="x", pad=2)

    # Shared bottom legend (one row of category swatches)
    # Use union of categories across the three panels, preserve order
    all_cats = []
    for c in GHG_CATS + FOS_CATS + WAT_CATS:
        if c not in all_cats:
            all_cats.append(c)
    handles = [Patch(facecolor=PALETTE[c], edgecolor="white", label=DISPLAY_NAME[c])
               for c in all_cats]
    fig.legend(handles=handles, loc="lower center",
               ncol=5, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.005),
               handlelength=1.1, handleheight=1.0, columnspacing=1.6,
               handletextpad=0.5, frameon=False)

    fig.savefig(f"{out_stem}.tiff", dpi=300)
    fig.savefig(f"{out_stem}.svg", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "outputs", "lca_4bar")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "Cotton_LCA_4bar")
    make_figure(stem)
    for ext in ("tiff", "svg"):
        p = f"{stem}.{ext}"
        print(f"wrote {p}  ({os.path.getsize(p)/1024:.1f} KB)")
