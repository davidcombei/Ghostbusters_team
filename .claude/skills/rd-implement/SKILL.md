---
name: rd-implement
description: The module contract for implementing an approved RTC-SDD lever plan as an isolated, additive package (SLS_setup/<bNwave>/ + scripts/<bN>_*) - scaffolded trainer copy, subclasses of the base code, default-off flags that reproduce the base bit for bit, checks.py (cpu / gpu / overfit), README flag table, exploop-compatible logs and run files, eval path for state_dict changes. Used by the rdloop IMPLEMENT / FIX steps (research-engineer) and interactively when the owner says "implement this lever in a new module", or invokes /rd-implement <cycle>.
disable-model-invocation: true
---

# Module contract (the b3wave precedent, codified)

The new module must let `python -u -m <module>.train <base argv>` reproduce the base run exactly, and
add each approved lever behind its own flags. Older experiments must stay reproducible: **no existing
file changes** (the supervisor hashes every file outside your scope after each step; any change blocks
the cycle).

## Layout (the supervisor scaffolds the first two)

| file | content |
|---|---|
| `SLS_setup/<module>/__init__.py` | docstring + `__version__` (scaffolded) |
| `SLS_setup/<module>/train.py` | copy of `SLS_setup/<base module>/train.py` + the `<module>: <version>` header line (scaffolded); you add the lever flags and wiring here |
| `SLS_setup/<module>/<area>.py` | lever code: subclasses / wrappers of the base classes (`b3wave.augment.RTCAugmenterB3`, `b3wave.model.ModelSLSB3`, `b3wave.data.SpoofAudioDatasetB3`, `b3wave.losses`, `b3wave.optim`, `utils.*`, `model.sls_model.ModelSLS`) - subclass, don't copy |
| `SLS_setup/<module>/checks.py` | `python -m <module>.checks cpu`, `gpu --ckpt <path>`, `overfit --lever <id>` |
| `SLS_setup/<module>/README.md` | what the module is, reproduction statement, flag table (flag, default, what it changes, file, check), telemetry, reserve notes |
| `scripts/<prefix>local_eval.py` (+ main_eval, profile) | only if an approved lever changes the state_dict (see below) |

Imports go one way: the module imports from `<base module>`, `model`, `utils`, `main_eval`; nothing old
imports the module. Run everything from `SLS_setup/` (`cd SLS_setup && python -m <module>.checks cpu`).

## Flags

- New argparse group named after the module (`b4 = parser.add_argument_group("B4 wave")`); names and
  defaults exactly as approved.
- **Default = base, bit for bit**: when a lever is off it must not draw random numbers (python `random`,
  numpy, torch, the augmenter's `rng`), add parameters or buffers, change shapes, dtypes or the order of
  operations, or change any log line. Gate the whole code path on the flag.
- Each lever prints ONE header line at startup, on or off (e.g. `AugPLC: off` / `AugPLC: on p=0.3
  loss=2..8%`); the plan's `expect_regex` must match it when on.
- Telemetry lines per epoch are fine (`Epoch N <MOD> ...`); never change the `Epoch N/M TrainLoss=...`
  or `Epoch N LocalWF1=...` formats.

## Exploop contract (the wave supervisor parses these)

- stdout ends with `Experiment saved to: <run dir>`; the run dir is `<out_path>/<track>_epoch<N>_bs<B>_<ts>`.
- `config.yaml` keeps `command` (= `" ".join(sys.argv)`), `args`, git fields, and gains
  `<module>_version` and `module_sha256` (sha256 over the module's `.py` files, sorted by name; also
  print `<module> sha256: <first 12>`), via `write_run_config(..., extra={...})`.
- `metrics.jsonl`, `logs/train.log`, `ckpt/epoch_<N>_dev_loss_<x>.pth` (EMA weights), `last_state.pt`,
  `--resume`, `--keep_topk_by_wf1`, `--keep_epochs`, `--track`, `--seed` behave as in the base.

## checks.py (print `PASS <name>` / `FAIL <name>: <reason>`, exit non-zero on any failure)

- `cpu` (you run it; CPU only, < ~5 min): argparse defaults of every new flag equal the plan; levers-off
  equivalence at the unit level (e.g. the augmenter with defaults draws exactly the base augmenter's
  random numbers on N synthetic clips - see `b3wave/checks.py::check_augmenter`; a head with the flag
  off has the base's state_dict keys); each lever on: runs, changes what it should, respects
  `heldout_reserve.assert_training_ops`; no reserved ops reachable.
- `gpu --ckpt <path>` (the supervisor runs it): load the base checkpoint into the new model with all
  flags off -> exact keys and identical logits to the base model on a fixed batch; each lever on ->
  forward + backward run, finite loss.
- `overfit --lever <id>` (levers with `params_added`): 60 steps on 16 clips -> loss drops >= 50 % and
  the new parameters receive gradients.

## State-dict changes (at most one approved lever)

All eval scripts build plain `ModelSLS` from `--ssl_name/--n_layers`. A lever whose checkpoints have
different keys needs: `scripts/<prefix>local_eval.py` (same CLI as `scripts/local_eval.py`, plus
`--run_config <run>/config.yaml` to rebuild the model with the run's flags; `--merge_list` must work),
the matching progress-scoring script, and `scripts/exploop/profiles/<prefix>local17.json` (a copy of
`sls_local17.json` whose GPU steps call them). With the flag off the checkpoints stay plain `ModelSLS`.

## Working rules

- CPU only (CUDA is hidden; never re-enable it). Never train, run GPU checks, start background jobs,
  install packages or use the network. A missing package -> `needs_user` in your output.
- Write only inside the module dir, `scripts/<prefix>*`, `rdloop/<C>/impl/`, `rdloop/<C>/scratch/`,
  `$TMPDIR`. Read anything (never progress scores).
- Keep `rdloop/<C>/impl/progress.md` current (done / left / decisions); a retry continues from it.
- Implement exactly the approved plan (`plan_agreed.json`); list every deviation with the reason.
- Before finishing: `python -m py_compile` your files, `python -m <module>.train --help` shows the
  flags, `python -m <module>.checks cpu` passes. Report the commands and results in `cpu_results`.
- In a FIX round address every listed finding / failure, say which in `addressed`, and keep the
  defaults reproducing the base.
