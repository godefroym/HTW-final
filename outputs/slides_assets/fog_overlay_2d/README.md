# Slide Assets: 2D Fog Overlay GIFs

These GIFs show the Dynamic Maze 2D/fog setting with the full maze visible for
presentation. Cells outside the current observation radius are tinted blue-grey
so the hidden fog-of-war region stays visible on slides.

Color key:

- red: agent
- yellow: goal
- orange: visited trail
- green/blue: open/closed stochastic gates
- blue-grey tint: currently outside the fog-of-war observation

Baseline files:

- `oracle_replan_positive_fullmaze_fog_overlay.gif`
- `memory_optimistic_positive_fullmaze_fog_overlay.gif`
- `fog_optimistic_negative_fullmaze_fog_overlay.gif`
- `fog_conservative_negative_fullmaze_fog_overlay.gif`
- `memory_conservative_negative_fullmaze_fog_overlay.gif`
- `random_negative_fullmaze_fog_overlay.gif`

JEPA/WM controller files:

- `repr_dist_positive_fullmaze_fog_overlay.gif`
- `learned_value_positive_fullmaze_fog_overlay.gif`
- `probe_pos_positive_fullmaze_fog_overlay.gif`
- `belief_value_positive_fullmaze_fog_overlay.gif`
- `jepa_subgoal_n2_positive_fullmaze_fog_overlay.gif`
- `jepa_subgoal_n4_positive_fullmaze_fog_overlay.gif`

The JEPA/WM GIFs are also duplicated in
`outputs/slides_assets/fog_overlay_2d_algorithms/` with a dedicated README and
summary. These are replayed from checkpoints, not retrained, and are not from
the 2.5D vision setting.
