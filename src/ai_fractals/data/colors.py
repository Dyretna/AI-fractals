# ai_fractals/data/colors.py

from matplotlib import colormaps

# -----------------------------------------------------------------------------
# Sorted colormaps from matplotlib registry
# -----------------------------------------------------------------------------
GREYS = [
    "Greys",
    "Grays",
    "gray",
    "grey",
    "gist_grey",
    "gist_gray",
    "gist_yerg",
    "gist_yarg",
    "binary",
    # reversed
    "Greys_r",
    "Grays_r",
    "gray_r",
    "grey_r",
    "gist_grey_r",
    "gist_gray_r",
    "gist_yerg_r",
    "gist_yarg_r",
    "binary_r",
]

SINGLE_COLOR_GRADIENTS = [
    "Blues",
    "Greens",
    "Oranges",
    "Purples",
    "Reds",
    "pink",
    "copper",
    # reversed
    "Blues_r",
    "Greens_r",
    "Oranges_r",
    "Purples_r",
    "Reds_r",
    "pink_r",
    "copper_r",
]

TWO_COLOR_GRADIENTS = [
    "BrBG",
    "BuGn",
    "BuPu",
    "GnBu",
    "OrRd",
    "PRGn",
    "PiYG",
    "PuBu",
    "PuOr",
    "PuRd",
    "RdBu",
    "RdGy",
    "RdPu",
    "YlGn",
    "Wistia",
    # reversed
    "BrBG_r",
    "BuGn_r",
    "BuPu_r",
    "GnBu_r",
    "OrRd_r",
    "PRGn_r",
    "PiYG_r",
    "PuBu_r",
    "PuOr_r",
    "PuRd_r",
    "RdBu_r",
    "RdGy_r",
    "RdPu_r",
    "YlGn_r",
    "Wistia_r",
]

THREE_COLOR_GRADIENTS = [
    "PuBuGn",
    "RdYlBu",
    "RdYlGn",
    "YlGnBu",
    "YlOrBr",
    "YlOrRd",
    # reversed
    "PuBuGn_r",
    "RdYlBu_r",
    "RdYlGn_r",
    "YlGnBu_r",
    "YlOrBr_r",
    "YlOrRd_r",
]

THEMED = [
    "magma",
    "inferno",
    "plasma",
    "viridis",
    "cividis",
    "twilight",
    "twilight_shifted",
    "turbo",
    "berlin",
    "managua",
    "vanimo",
    "Spectral",
    "afmhot",
    "autumn",
    "bone",
    "brg",
    "bwr",
    "cool",
    "coolwarm",
    "copper",
    "cubehelix",
    "gist_earth",
    "gist_heat",
    "gist_ncar",
    "gist_rainbow",
    "gist_stern",
    "gnuplot",
    "gnuplot2",
    "hot",
    "hsv",
    "jet",
    "nipy_spectral",
    "ocean",
    "prism",
    "rainbow",
    "seismic",
    "spring",
    "summer",
    "terrain",
    "winter",
    "CMRmap",
    # Reversed
    "magma_r",
    "inferno_r",
    "plasma_r",
    "viridis_r",
    "cividis_r",
    "twilight_r",
    "twilight_shifted_r",
    "turbo_r",
    "berlin_r",
    "managua_r",
    "vanimo_r",
    "Spectral_r",
    "afmhot_r",
    "autumn_r",
    "bone_r",
    "brg_r",
    "bwr_r",
    "cool_r",
    "coolwarm_r",
    "copper_r",
    "cubehelix_r",
    "gist_earth_r",
    "gist_heat_r",
    "gist_ncar_r",
    "gist_rainbow_r",
    "gist_stern_r",
    "gnuplot_r",
    "gnuplot2_r",
    "hot_r",
    "hsv_r",
    "jet_r",
    "nipy_spectral_r",
    "ocean_r",
    "prism_r",
    "rainbow_r",
    "seismic_r",
    "spring_r",
    "summer_r",
    "terrain_r",
    "winter_r",
    "CMRmap_r",
]

# psychedelic
DISCRETE_GRADIENTS = [
    "flag",
    "Accent",
    "Dark2",
    "Paired",
    "Pastel1",
    "Pastel2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    # reversed
    "flag_r",
    "Accent_r",
    "Dark2_r",
    "Paired_r",
    "Pastel1_r",
    "Pastel2_r",
    "Set1_r",
    "Set2_r",
    "Set3_r",
    "tab10_r",
    "tab20_r",
    "tab20b_r",
    "tab20c_r",
]

# ------------------------------------------------------------------------------
# Creating Curated colormap
# Keep:
#   GRADIENT_COMBINATIONS
#   THEMED
#   DISCRETE_GRADIENTS
#
# Filter out:
#   GREYS
#   SINGLE_COLOR_GRADIENTS
#   TWO_COLOR_GRADIENTS
#   "custom filter"
# ------------------------------------------------------------------------------
CUSTOM_OUT = [
    "hot",
    "gist_heat",
    "cividis",
    "viridis",
    # reversed
    "hot_r",
    "gist_heat_r",
    "cividis_r",
    "viridis_r",
]

# Combine all redundant
OUT_FILTERED_RGB = [
    *GREYS,
    *SINGLE_COLOR_GRADIENTS,
    *TWO_COLOR_GRADIENTS,
    *CUSTOM_OUT,
]

OUT_FILTERED_SHORELINES = [
    *GREYS,
    *SINGLE_COLOR_GRADIENTS,
    *TWO_COLOR_GRADIENTS,
    *THREE_COLOR_GRADIENTS,
    *CUSTOM_OUT,
]

CURATED_COLORMAPS = [c for c in list(colormaps) if c not in OUT_FILTERED_RGB]
CURATED_SHORELINE_COLORMAPS = [
    c for c in list(CURATED_COLORMAPS) if c not in OUT_FILTERED_SHORELINES
]


if __name__ == "__main__":
    print("\n Curated colormaps: ", CURATED_COLORMAPS)
    print("\n Curated Shoreline colormaps: ", CURATED_SHORELINE_COLORMAPS)
