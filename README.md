# gpumd-nep89-workflow# GPUMD-NEP89 Active-Learning Workflow

This repository provides a clean, GitHub-ready workflow for using the GPUMD NEP89 potential to sample structures, prepare DFT single-point calculations, evaluate NEP89 predictions, fine-tune a NEP potential, and run production GPUMD simulations.

The original Word tutorial has been converted into this README and the repository has been reorganized so that each folder maps to one stage of the workflow.

## Repository layout

```text
gpumd-nep89-workflow/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- data/
|   |-- sampling/              # Raw and selected structures from GPUMD sampling
|   |-- prediction/            # Training set used for NEP89 prediction checks
|   `-- finetuning/            # Training set used for NEP fine-tuning
|-- docs/
|   `-- workflow.md            # Condensed workflow notes
|-- models/
|   |-- nep89/                 # Base NEP89 potential
|   `-- finetuned/             # Fine-tuned NEP potential
|-- scripts/
|   |-- prepare_temperature_runs.py
|   |-- sample_structures.py
|   |-- convert_extxyz_to_dft_inputs.py
|   |-- plot_nep_results.py
|   `-- run_gpumd_jobs.sh
`-- workflow/
    |-- 01_sampling/           # GPUMD sampling inputs
    |-- 02_nep89_prediction/   # NEP prediction inputs
    |-- 03_nep_finetuning/     # NEP fine-tuning inputs
    `-- 04_production/         # Production GPUMD simulation inputs
```

## Prerequisites

Install and configure the following tools before running the workflow:

- GPUMD with NEP support
- GPUMDkit
- Python 3.9 or newer
- Python packages listed in `requirements.txt`
- ASE for reading and writing extended XYZ and VASP POSCAR files
- Optional: VASPkit for VASP input preparation
- Optional: dpdata and ABACUS tools for ABACUS input preparation

A typical GPUMDkit environment setup is:

```bash
export GPUMD_path=/path/to/GPUMD
export GPUMDkit_path=/path/to/GPUMDkit
export PATH=${GPUMDkit_path}:${PATH}
source ${GPUMDkit_path}/Scripts/utils/completion.sh
chmod +x gpumdkit.sh
gpumdkit.sh -h
```

If your compiler environment requires a newer GCC toolchain, enable it before compiling GPUMD. For example:

```bash
source /opt/rh/devtoolset-9/enable
```

## Stage 1: Structure sampling with NEP89

Use a small `model.xyz` system and the base NEP89 potential to generate representative structures.

```bash
cd workflow/01_sampling
gpumd
```

The sampling input uses `run.in`. The key structure format requirement is that the second line of `model.xyz` should define periodic boundary conditions, lattice vectors, and per-atom properties. For example:

```text
pbc="T T T" lattice="22.0 0.0 0.0 0.0 22.0 0.0 0.0 0.0 22.0" Properties=species:S:1:pos:R:3:group:I:1
```

Here, `species:S:1` stores the element symbol, `pos:R:3` stores the three Cartesian coordinates, and `group:I:1` stores a group ID. The group column can be omitted if no group-based post-processing is needed.

For batch temperature folders, run this script in a folder containing `run.in` and `model.xyz`:

```bash
python ../../scripts/prepare_temperature_runs.py -T 300 500 700 900 1100 1300 1500
```

## Stage 2: Select structures for DFT labeling

After GPUMD sampling, select representative frames from the generated extended XYZ trajectory.

```bash
cd workflow/01_sampling
cp ../../data/sampling/dump.xyz .
python ../../scripts/sample_structures.py select pca 100
```

This produces files such as `select_100.xyz` and `abandon_100.xyz`, together with PCA visualization plots.

Convert selected structures to VASP or ABACUS inputs:

```bash
python ../../scripts/convert_extxyz_to_dft_inputs.py select_100.xyz 100 vasp
```

For VASP, the script expects an `INCAR-scf` file in the working directory and writes calculation folders under `train_folders/`. After single-point calculations are complete, use GPUMDkit to assemble a multi-frame NEP dataset:

```bash
gpumdkit.sh
# Select: 101 -> 1 -> .
```

The generated dataset is typically located at:

```text
NEPdataset-multiple_frames/NEP-dataset.xyz
```

Copy this file and rename it to `train.xyz` for the prediction or fine-tuning stages.

## Stage 3: Evaluate base NEP89 prediction quality

Use the generated `train.xyz` to check the base NEP89 prediction quality for energy, force, and virial.

```bash
cd workflow/02_nep89_prediction
nep
python ../../scripts/plot_nep_results.py
```

The `nep.in` file uses `prediction 1`, so the run evaluates the existing `nep.txt` potential instead of training a new one.

## Stage 4: Fine-tune the NEP potential

If the base NEP89 model performs poorly for the target chemical space, fine-tune it with the labeled dataset.

```bash
cd workflow/03_nep_finetuning
nep
python ../../scripts/plot_nep_results.py
```

The provided `nep.in` uses:

```text
fine_tune ../../models/nep89/nep.txt nep89_20250409.restart
```

Before running fine-tuning, place the matching NEP89 restart file named `nep89_20250409.restart` in `workflow/03_nep_finetuning/`, or edit the path to point to your local restart file.

The fine-tuning stage produces a new `nep.txt`, which can be copied into `models/finetuned/` and used for production simulations.

## Stage 5: Production GPUMD simulation

Use the fine-tuned `nep.txt` to simulate a larger production system.

```bash
cd workflow/04_production
gpumd
```

The provided production input performs minimization, NPT equilibration, NVT simulation, trajectory dumping, and MSD calculation.

## Typical end-to-end command summary

```bash
# 1. Sampling
cd workflow/01_sampling
gpumd
python ../../scripts/sample_structures.py select pca 100
python ../../scripts/convert_extxyz_to_dft_inputs.py select_100.xyz 100 vasp

# 2. NEP89 prediction check
cd ../02_nep89_prediction
nep
python ../../scripts/plot_nep_results.py

# 3. NEP fine-tuning
cd ../03_nep_finetuning
nep
python ../../scripts/plot_nep_results.py

# 4. Production simulation
cd ../04_production
gpumd
```

## Notes on large files

This repository currently keeps the provided example `nep.txt`, `train.xyz`, `dump.xyz`, and selected structures so the workflow remains self-contained. For a public GitHub release, consider using Git LFS for large model and trajectory files.

## Troubleshooting

- If `sample_structures.py` cannot calculate descriptors, check that `NepTrain` is installed and that the `nep_name` path inside the script points to a valid NEP89 potential.
- If POSCAR generation fails, verify that ASE can read your extended XYZ file and that the first line of each frame is the atom count.
- If `nep` cannot start fine-tuning, verify that both the base NEP89 `nep.txt` and its matching restart file are available.
- If GPUMD cannot find the potential file, update the `potential` line in `run.in` to point to the correct `nep.txt` path.
