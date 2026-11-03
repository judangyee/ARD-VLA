# Paper figures

## Fig.4 — `paper/scripts/make_fig4.py`

2-panel analysis figure from evaluation rollout logs.

1. Log rollouts with `RolloutLogger` (`lerobot.policies.smolvla.eval_logging`) during
   `run_closed_loop(..., per_step=logger.per_step, track_role_split=True)`, then
   `logger.save("rollout_logs/<method>_<episode_id>.npz")` per episode.
2. Generate the figure:

   ```bash
   python paper/scripts/make_fig4.py rollout_logs/*.npz --out paper/figures/fig4_analysis
   ```

   Writes `paper/figures/fig4_analysis.{pdf,svg,png}`. Panel (a) needs `contact` +
   `stabilizer_object_pos`/`ee_pos` in the logs; panel (b) needs `r_stab_norm`/`r_act_norm`
   (only present for `method="ard_vla"` logs with `track_role_split=True`). Missing data is
   skipped, never fabricated — a panel shows a "no data" placeholder instead of fake numbers.

   For a smoke test on dummy data, pass `--watermark "SMOKE TEST"` and an `--out` path
   outside `paper/figures/` (e.g. a scratch directory) — real figures in `paper/figures/`
   must only ever come from real evaluation logs.
