# Track 7: Learned cost / value for planning

This folder tracks the Two Rooms implementation for the hackathon Track 7 story:
replace the hand-crafted latent-distance MPC cost with a TD-MPC-style learned
goal-conditioned value trained on the world model's own rollouts.

## Current status

- [x] `GoalValueHead` exists in `eb_jepa/state_decoder.py`.
- [x] `LearnedValueMPCObjective` exists in `eb_jepa/planning.py`.
- [x] `learned_value` is registered in `objective_name_map`.
- [x] Value TD(0) training is wired in `examples/ac_video_jepa/main.py`.
- [x] Two Rooms value-training config added.
- [x] Two Rooms baseline/value planning configs added.
- [x] Run script added for the controlled comparison.
- [x] Summary script added for CSV/LaTeX/plot output after eval.
- [ ] Launch the value-head training on Dalia from a stable Two Rooms WM checkpoint.
- [ ] Evaluate `repr_dist` vs `learned_value` at matched MPPI compute.
- [ ] Aggregate results into a table / Pareto plot.
- [ ] Pick representative GIFs for the final slides.

## Main protocol

Use the same world model, same MPPI settings, same eval config, and change only:

```text
planner.planning_objective.objective_type = repr_dist
planner.planning_objective.objective_type = learned_value
```

The value head is trained with the world model frozen. This keeps the comparison
clean: the experiment asks whether a learned value is a better planner cost than
latent distance, not whether a larger or differently trained world model is better.

## Maze README lessons we keep

The Maze README is treated as supporting evidence, not as the main Track 7
protocol. The important takeaways are:

- freeze a proven world model first; Maze co-training made the latent less useful
  for wall-aware low-level control;
- compare costs with all planner scaffolding matched;
- avoid using A* as the main planner signal for the Track 7 claim;
- report the horizon limitation honestly, because greedy-global Maze stayed at
  0% even with a learned value.

This is why the main run is Two Rooms `repr_dist` vs `learned_value`, while the
Maze value/hierarchy numbers are kept as an ambitious stress-test appendix.

## Files

- `HTW_TRACK7_VALUE_PLAN.tex`: explanation and jury-facing protocol.
- `run_two_rooms_track7.sh`: staged launcher for value training and matched evals.
- `summarize_two_rooms_track7.py`: turns eval folders into CSV, LaTeX table, and plot.
- `../cfgs/train/two_rooms/train_value.yaml`: frozen-WM value training.
- `../cfgs/eval/two_rooms/eval_track7.yaml`: 32-episode normal-level eval.
- `../cfgs/planning/two_rooms/planning_mppi_repr_track7*.yaml`: distance baselines.
- `../cfgs/planning/two_rooms/planning_mppi_value_track7*.yaml`: learned-value evals.

## Example command

```bash
bash examples/ac_video_jepa/track7_value/run_two_rooms_track7.sh \
  /path/to/two_rooms/latest.pth.tar \
  /path/to/output/track7_two_rooms
```

The script trains the value head once, then runs matched evaluations for:

- `repr_dist` at 50 samples;
- `learned_value` at 50 samples;
- `repr_dist` at 200 samples;
- `learned_value` at 200 samples.

After the runs:

```bash
python examples/ac_video_jepa/track7_value/summarize_two_rooms_track7.py \
  /path/to/output/track7_two_rooms
```

## What remains

The main missing piece is a launched run on Dalia using a stable Two Rooms
checkpoint. After that, the results should be summarized as success rate, final
distance, and average episode time. The final slide should lead with the matched
comparison and keep the maze/A* result as an ambitious stress-test appendix.
