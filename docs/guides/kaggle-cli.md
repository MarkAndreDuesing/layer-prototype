# Kaggle CLI workflow

> **Role: guide.** Step by step; how we push and collect every Kaggle trial.

1. Install the CLI in its own venv and sign in:
   ```bash
   python -m venv kagglecli
   ```
   ```bash
   kagglecli\Scripts\pip install kaggle
   ```
   Create a token on https://www.kaggle.com/settings (API → Create New Token) and save it as `%USERPROFILE%\.kaggle\kaggle.json`.
2. Generate a notebook with its generator (each writes `prototype/kaggle/upload_<name>/` with the `.ipynb` and `kernel-metadata.json`):
   ```bash
   fastsdcpu\env\Scripts\python.exe prototype\tools\make_qwen21_rgba_trial.py
   ```
3. Push it from inside its folder (the CLI has a path bug otherwise):
   ```bash
   cd prototype\kaggle\upload_qwen21_rgba_trial && python -m kaggle.cli kernels push -p .
   ```
4. Check status, then download outputs (use `--file-pattern` to skip cloned repos):
   ```bash
   python -m kaggle.cli kernels status markandreduesing/qwen21-rgba-trial
   ```
   ```bash
   python -m kaggle.cli kernels output markandreduesing/qwen21-rgba-trial -p prototype/kaggle/results_qwen21_vN
   ```
5. Record the run: add a row to `prototype/kaggle/benchmarks.csv`, run `python prototype/tools/build_benchmarks.py`, update the factsheet and [../status.md](../status.md), append to [../log.md](../log.md).

## Rules learned
- `machine_shape: NvidiaTeslaT4` gives 2× T4. **At most 2 batch GPU sessions at once**; a third push fails with "Maximum batch GPU session count of 2 reached".
- Notebooks of about 1.5 MB are rejected (HTTP 400): embed images as JPEG or attach data via `dataset_sources` / `kernel_sources`.
- Give every trial a SIGALRM time cap (25 min) so a stall still ends with a log; cancelled runs keep no output.
- Before re-pushing a stalled run, check kaggle.com/work/code for an outage banner.
- Smoke-test with 2 steps before a full run; verify locally with tiny models where possible.
- TPU via CLI: `machine_shape: TpuV5E8` is stored but batch runs got no TPU (see [../research/hosting.md](../research/hosting.md#tpu)).
