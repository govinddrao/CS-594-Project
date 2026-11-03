# Solar Forecasting for Dynamic Pricing and Energy Allocation

**CS 594 Project — Govind Rao**

This project investigates how forecasts of solar generation can support
electricity pricing and allocation across microgrids with battery storage.
The planned decision model is a weakly coupled Markov decision process
(WCMDP): each microgrid has local battery dynamics and customer demand,
while electricity allocations share a supply constraint.

## Current progress

- Implemented preprocessing for the SKIPP’D solar forecasting dataset.
- Processed an initial 60-day subset into forecasting examples.
- Implemented a microgrid simulator and three baseline policies.
- Generated baseline results using recorded solar generation.
- CNN training and dynamic WCMDP policy methods remain to be implemented.

## Repository contents

| File | Description |
|---|---|
| `preprocess_skippd.py` | Validates observations, constructs forecasting examples, creates chronological splits, and provides a lazy data loader |
| `prepared_skippd.npz` | Prepared example indices, targets, split assignments, and normalization statistics |
| `simulate_microgrids.py` | Simulates demand, electricity allocations, batteries, and financial outcomes |
| `rollouts.csv` | Step-by-step results for each microgrid and policy |
| `episode_summary.csv` | Results for each policy and simulation episode |
| `policy_summary.csv` | Aggregate baseline results |
| `config.json` | Parameters and assumptions used for the saved simulation run |

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/govinddrao/CS-594-Project.git
cd CS-594-Project

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas h5py
```

PyTorch is not required for preprocessing or simulation. It will be
needed for the planned CNN training stage.

## Dataset

We use [SKIPP’D](https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset),
which contains sky images and photovoltaic (PV) power measurements
collected at Stanford University during 2017–2019.

Download the processed benchmark from the
[Stanford dataset page](https://purl.stanford.edu/dj417rh1007).
Create a local `data` directory containing:

- `2017_2019_images_pv_processed.hdf5`
- `times_trainval.npy`
- `times_test.npy`

The large dataset files must be downloaded separately.

### Preprocessing

Inspect the dataset structure:

```bash
python3 preprocess_skippd.py --data-dir data --inspect
```

Prepare the earliest 60 available days:

```bash
python3 preprocess_skippd.py \
  --data-dir data \
  --trusted-timestamps \
  --max-days 60 \
  --output prepared_local.npz
```

The separate output name avoids overwriting the prepared index already
included in the repository. The timestamp option should only be used
with trusted files, such as the official dataset files.

The script:

1. Checks array alignment, timestamps, power readings, and image values.
2. Constructs 16-frame histories covering minutes `t−15` through `t`.
3. Uses power at `t+15` minutes as the prediction target.
4. Samples examples every five minutes within uninterrupted segments.
5. Excludes windows crossing gaps or day boundaries.
6. Creates chronological splits using approximately 70%, 15%, and 15%
   of the days for training, validation, and testing.
7. Calculates power normalization statistics from training targets only.

The loader scales image pixels to `[0, 1]`, standardizes power readings,
and adds sine/cosine time-of-day features. Images remain in the original
HDF5 file and are loaded on demand.

These chronological splits replace the authors’ original benchmark
partition.

### Initial preprocessing results

| Quantity | Count |
|---|---:|
| Observations in the 60-day subset | 47,069 |
| Training examples | 4,365 |
| Validation examples | 853 |
| Test examples | 1,028 |
| Total forecasting examples | 6,246 |

The target currently represents power 15 minutes ahead, not total
energy generated during the following interval.

## Microgrid simulation

The simulator represents an electricity retailer serving three
microgrids. Each microgrid has price-responsive customer demand and
a local battery.

At each one-minute step, a policy selects customer prices and electricity
allocations. Demand is then realized, consumption and battery levels
update, and a reward is calculated.

The model includes:

- Recorded solar generation and an assumed fixed electricity contract.
- Independent local demand shocks.
- Battery capacity, charging/discharging limits, and efficiency losses.
- Revenue, procurement costs, battery wear, and unmet-demand penalties.
- A shared constraint on total electricity allocations.

Demand parameters, costs, and battery specifications are illustrative
simulation assumptions, not estimates from real customers.

### Run with recorded solar generation

After creating `prepared_local.npz`, run:

```bash
python3 simulate_microgrids.py \
  --prepared prepared_local.npz \
  --data-dir data \
  --trusted-timestamps \
  --split test \
  --days 3 \
  --output simulation_results_local
```

The simulator uses the prepared index to select days, then reads
consecutive PV measurements from the original dataset.

Each episode is a contiguous daytime segment. Recording gaps create
separate episodes, each starting with half-full batteries. Overnight
operation is not simulated.

### Run without downloading the dataset

```bash
python3 simulate_microgrids.py \
  --synthetic \
  --days 3 \
  --output simulation_results_synthetic
```

### Baseline policies

| Policy | Description |
|---|---|
| `fixed` | Uses a fixed price and allocates electricity proportionally to anticipated needs |
| `myopic` | Optimizes an approximate deterministic one-step objective over discretized prices and allocations |
| `explore` | Tries random prices and records resulting demand |

Policies are evaluated using common solar trajectories and latent demand
shocks. Actual demand differs across policies because prices differ.

The current baselines observe the current interval’s supply and know
the assumed demand parameters. They do not use CNN forecasts or observe
future demand and generation.

### Outputs

Each simulation run creates:

- `rollouts.csv`
- `episode_summary.csv`
- `policy_summary.csv`
- `config.json`

Output directories must be new; choose another `--output` name when
rerunning an experiment.

The prepared index stores the source HDF5 path. If that file has moved,
pass its new path using `--h5`.

## Saved baseline results

The committed results contain 21 recording-segment episodes from three
selected test days, using seed 42.

| Policy | Total profit ($) | Total unmet demand (kWh) |
|---|---:|---:|
| Fixed | 151.07 | 87.33 |
| Myopic | 286.31 | 8.01 |
| Exploratory | 181.05 | 53.11 |

These are preliminary simulator results under one parameter setting.
They do not establish the value of CNN forecasts or the performance
of a dynamic WCMDP policy.

Policies can induce different consumption levels. Interpretation should
also consider requested and served demand, final battery levels, and
the absence of a terminal battery salvage payment.

## Planned work

- Visually inspect representative forecasting examples.
- Train a CNN using sky images, past PV measurements, and time features.
- Compare forecasting accuracy and regularization choices.
- Connect forecasts and supply uncertainty to the decision model.
- Investigate Lagrangian relaxation, feasibility network relaxation,
  and delayed allocation for WCMDP policy construction.
- Evaluate profit, unmet demand, computational cost, and robustness
  across multiple seeds and parameter settings.

## Dataset reference

Nie, Y., Li, X., Scott, A., Sun, Y., Venugopal, V., and Brandt, A.
(2023). *SKIPP’D: A SKy Images and Photovoltaic Power Generation Dataset
for Short-term Solar Forecasting*. Solar Energy, 255, 171–179.

Dataset documentation:
https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset
