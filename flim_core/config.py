"""
Tunable constants shared across every analysis plugin and flim_core module.

Moved out of the old flimera_flim_analysis.py (which used to be both "the main
script" and "the shared config module" at once) so every plugin depends on the
same neutral place for these instead of on one specific analysis.
"""

DEFAULT_THRESHOLD = 400

TAIL_START_OFFSET = 50

TAU_SHORT_BOUNDS  = (0.5, 2.0)

TAU_LONG_BOUNDS   = (2.8, 8)

TAU_MONO_BOUNDS   = (TAU_SHORT_BOUNDS[0], TAU_LONG_BOUNDS[1])

DISPLAY_ASPECT    = 2

# No longer used by default: flim_core/plotting.py auto-scales each result
# plot's colorbar and histogram range from that run's own fitted values
# (see plotting._auto_vlim), so results look right for any dye without
# retuning this. Kept only as an OPTIONAL explicit override -- pass it as
# plot_results(..., vlim=TAU_LONG_VLIM) if you specifically want several
# files' plots to share one fixed color/axis scale for a side-by-side
# comparison.
TAU_LONG_VLIM     = (2.8, 3.8)

MIN_NEIGHBORS     = 2

BIN_SIZE          = 1

IRF_SHIFT_SEARCH = 80  # max bins searched either direction when aligning
                        # an IRF's timing to the data's
