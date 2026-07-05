# LookCloser Frequency Grid Schedule Sweep

## What was tested

Attempted to sweep the frequency ladder settings after the three-seed baseline:

- `max_res_base=1024`
- planned next: `max_res_base=2048`, `max_res_base=4096`, `num_frequency_levels=12`, `num_frequency_levels=16`

The existing HD frequency maps were generated with:

```text
min_res=16
max_res=8192
n_levels=16
```

## Results

The sweep was stopped before producing metrics.

| Candidate | Seed | Result |
|---|---:|---|
| `max_res_base=1024` | 42 | Failed before training. |
| `max_res_base=1024` | 43 | Auto-started before stop; same metadata mismatch. |

Failure:

```text
ValueError: Frequency metadata ... does not match the model frequency grid:
metadata(min_res=16.0, max_res=8192.0, n_levels=16) vs
grid(min_res=16.0, max_res=4096.0, n_levels=16).
```

## Insights

The schedule axes are not independent hyperparameters with scalar-resolution frequency maps. Changing `max_res_base`, `max_res`, or `num_frequency_levels` changes the mapping between scalar 2D frequencies and model grid levels, so those candidates require regenerating frequency maps for the same schedule.

For the initial internal Frequency Grid optimization, keep the preprocessing-compatible schedule fixed:

```text
min_res=16
max_res_base=2048
num_frequency_levels=16
```

Continue with schedule-independent settings: grid resolution, update interval, update batch size, and fallback frequency level.
