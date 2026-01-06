## Naming rules for multi-tenant simulation results

Multi-tenant simulation results are stored under `Data/Evaluation/<cluster_type>/`, with separate subfolders per cluster type, for example:

- `Data/Evaluation/H100_Real/H100_26H100_27H100_28H100_29/`
- `Data/Evaluation/H100_Real/Het-4Mix/` (if enabled)

Within each `cluster_type` subdirectory, the CSV naming rule is:

`MTS_{random_seed}RS_{num_train_samples}TD_{contention_mode}CM_{repeat_num}RN.csv`

Field meanings:

- **MTS_{random_seed}RS**: `random_seed` from top-level `config.random_seed`
- **{num_train_samples}TD**: Training sample count from `training.num_train_samples`
- **{contention_mode}CM**: Multi-tenant contention mode from `evaluation.multi_tenant.contention_mode`
- **{repeat_num}RN**: Multi-tenant simulation repetitions from `evaluation.multi_tenant.repeat_num`

Example:

- `MTS_1111RS_500TD_commonCM_100RN.csv`

Meaning:

- Random seed `random_seed = 1111`
- Training samples `num_train_samples = 500`
- Contention mode `contention_mode = common`
- Multi-tenant simulation repetitions `repeat_num = 100`


