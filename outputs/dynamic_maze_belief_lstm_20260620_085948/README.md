# Dynamic Maze BeliefLSTM Value Run

Source run on Dalia:

```text
$EBJEPA_WORK/runs/dynamic_maze_belief_lstm_20260620_085948
```

This run freezes the bootstrap JEPA world model and trains:

```text
pooled(z_t), action_{t-1} -> BeliefLSTM(h=256) -> h_t
V(h_t, z_goal) -> scalar value
```

It stays A*-free for data collection and closed-loop planning.

## Result

| objective | success | mean steps | blocked moves | final manhattan |
|---|---:|---:|---:|---:|
| `belief_value` | 73.4% | 271.8 | 100.5 | 6.8 |

For comparison on the previous bootstrap run:

| objective | success | mean steps | blocked moves | final manhattan |
|---|---:|---:|---:|---:|
| `learned_value` | 70.3% | 289.5 | 88.6 | 7.0 |
| `repr_dist` | 75.0% | 244.9 | 85.1 | 4.1 |

Interpretation: recurrent memory improves the learned value objective, but the
first implementation still does not beat representation distance. The next
iteration should reduce blocked moves, likely by feeding moved/blocked feedback
or a compact visited-map signal into the belief state.

## Files

- `00_belief_value/metrics.tsv`: training loss per epoch.
- `01_eval_belief_value/belief_value_eval.json`: aggregate eval metrics.
- `01_eval_belief_value/*.gif`: first six evaluation episodes.
- `slurm_logs/`: job logs for Slurm job `76388`.
