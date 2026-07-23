"""
Single source of truth for per-element colours across the 03_SURF_HETERO
campaign: the matplotlib figures (via hop_screen.SPECIES_COLOR) and the
OVITO/Blender structure renders (via prod_structure_movie.py) both import from
here, so a render can never drift out of sync with the plot beside it.

Pure stdlib on purpose -- prod_structure_movie.py runs in the separate
`ovito-render` conda env, which has no matplotlib/numpy of this project's vintage.

2026-07-16: user-specified exact hex codes. The previous campaign palette is kept
below as SPECIES_COLOR_LEGACY; assign it to SPECIES_COLOR to revert every figure
and render in one edit.
"""

SPECIES_COLOR = {
    "Ba": "#1EEF2C",     # green
    "Ti": "#78CAFF",     # sky blue
    "O":  "#FE0300",     # red
    "Dy": "#3106FC",     # blue
    "Mg": "#FB7B15",     # orange
}

SPECIES_COLOR_LEGACY = {
    "Ba": "#2CA02C", "Ti": "#56B4E9", "O": "#D62728",
    "Dy": "#4338CA", "Mg": "#FF7F0E",
}

# Display radii (A) for the structure renders only -- visual, not physical.
# Cations big enough to read against the O sublattice; O kept small so the
# dopants stay visible inside the perovskite. io_mesh_atomic's own radii are
# much larger (Ba 1.98, Dy 1.59) and render as one solid mass of touching balls.
SPECIES_RADIUS = {"Ba": 1.35, "Ti": 0.90, "O": 0.55, "Dy": 1.25, "Mg": 0.85}

# io_mesh_atomic names every object and material after the FULL element name
# ("Barium_ball", material "Barium") -- symbol lookups silently miss and the
# atom keeps the addon's default colour, so the Blender side keys off these.
ELEMENT_FULLNAME = {"Ba": "Barium", "Ti": "Titanium", "O": "Oxygen",
                    "Dy": "Dysprosium", "Mg": "Magnesium"}


def rgb(sp, default=(0.5, 0.5, 0.5)):
    """'#3106FC' -> (0.192, 0.024, 0.988) floats, for OVITO/Blender."""
    h = SPECIES_COLOR.get(sp)
    if h is None:
        return default
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
