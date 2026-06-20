# Slide Assets: Dynamic Maze Algorithm Fog Overlay GIFs

These GIFs replay trained JEPA/WM controllers and render the full maze for
presentation. Cells outside the current observation radius are tinted
blue-grey so the hidden fog-of-war region stays visible on slides.

Color key:

- red: agent
- yellow: goal
- orange: visited trail
- green/blue: open/closed stochastic gates
- blue-grey tint: currently outside the fog-of-war observation

Generated algorithms:

- `repr_dist`: `repr_dist_positive_fullmaze_fog_overlay.gif` (success, moves=56, blocked=32, final_manhattan=0)
- `learned_value`: `learned_value_positive_fullmaze_fog_overlay.gif` (success, moves=60, blocked=39, final_manhattan=0)
- `probe_pos`: `probe_pos_positive_fullmaze_fog_overlay.gif` (success, moves=172, blocked=95, final_manhattan=0)
- `belief_value`: `belief_value_positive_fullmaze_fog_overlay.gif` (success, moves=60, blocked=39, final_manhattan=0)
- `jepa_subgoal_n2`: `jepa_subgoal_n2_positive_fullmaze_fog_overlay.gif` (success, moves=56, blocked=48, final_manhattan=0)
- `jepa_subgoal_n4`: `jepa_subgoal_n4_positive_fullmaze_fog_overlay.gif` (success, moves=56, blocked=48, final_manhattan=0)

Notes:

- The controllers are replayed from existing checkpoints; this is not a
  training run.
- A* is not called in the action loop for these JEPA/WM controllers. For
  subgoal GIFs, A* is used only to report the initial budget reference,
  matching the existing evaluation script.
