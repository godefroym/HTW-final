# Dynamic Maze Bootstrap WM Run

Source run on Dalia:

```text
$EBJEPA_WORK/runs/dynamic_maze_bootstrap_20260620_051011
```

This run trains the dynamic-maze JEPA/value setup without A* trajectories:

- data collection policy: `local_goal_frontier`;
- `layout_use_astar_filter: false`;
- value objective: `multi_horizon_td`;
- Slurm job: `75812`;
- runtime: `01:32:25` on one B200;
- eval: 64 episodes, lookahead 4, fixed 800-move budget.

## Results

| objective | success | mean steps | blocked moves | final manhattan |
|---|---:|---:|---:|---:|
| `learned_value` | 70.3% | 289.5 | 88.6 | 7.0 |
| `probe_pos` | 65.6% | 323.2 | 84.7 | 6.8 |
| `repr_dist` | 75.0% | 244.9 | 85.1 | 4.1 |

The learned value head is useful but not yet the best objective: `repr_dist`
still has the strongest first-run score. This is the main next optimization
target for Track 7.

## Files

- `metrics.tsv`: aggregate table.
- `01_eval_learned_value/`: learned value objective, JSON + first GIFs.
- `01_eval_probe_pos/`: position-probe objective, JSON + first GIFs.
- `01_eval_repr_dist/`: latent-distance objective, JSON + first GIFs.
- `slurm_logs/`: job stdout/stderr used to audit training and eval.
