# Deep reinforcement learning optimizes adaptive immunotherapy sequencing through tumor-immune coevolution modeling

CoEvo-DRL couples a ten-variable tumor–immune ordinary differential equation system to a phase-aware Proximal Policy Optimization agent. The policy selects anti-PD-1, anti-CTLA-4, dual checkpoint blockade, checkpoint inhibitor plus chemotherapy, or a treatment holiday at 21-day clinical decision points over a 24-month horizon.

## Scientific scope

The state contains sensitive and resistant tumor cells, effector CD8+ T cells, regulatory T cells, dendritic cells, neoantigen immunogenicity, PD-L1 expression, checkpoint inhibitor concentration, chemotherapy concentration, and tumor burden. Five derived variables encode neoantigen diversity, immune escape velocity, the Treg-to-effector ratio and its derivative, and normalized tumor growth. The reward combines burden change, active-treatment burden, and a Treg-ratio-modulated holiday term.

The supplied manuscript does not include the supplementary tables it cites for complete ODE parameters, full baseline hyperparameters, or compute specifications. Values absent from the manuscript are exposed as typed configuration fields in `ODEParameters`; they are biological initialization values and must be replaced with the study's approved Supplementary Table S1 values before reporting numerical results. The manuscript's explicit PPO values are retained unchanged.

## Installation

Create the pinned environment with either:

```bash
python -m venv .venv
. .venv/bin/activate
pip install .
```

```bash
conda env create -f environment.yml
conda activate coevo-drl
pip install --no-deps .
```

The container uses PyTorch 2.5.1 with CUDA 12.4:

```bash
docker build -t coevo-drl .
```

## Data

Verified canonical access points are collected in `dataset_links.txt`. SU2C-MARK is controlled through dbGaP and requires an approved request. MSK-IMPACT data are accessed through cBioPortal under the terms attached to the selected study. TCGA-LUAD contains both open and controlled files through the Genomic Data Commons. No patient data are bundled.

Input patient tables use the columns `patient_id`, `tumor_mutational_burden`, `neoantigen_count`, `cd8_fraction`, `treg_fraction`, `dendritic_fraction`, `pdl1_expression`, `survival_days`, and `event`. Cohort manifests should be hashed after authorized download because controlled releases and portal exports can change.

## Training

The manuscript training budget is one million environment steps, 5,000 virtual training patients, PPO updates every 2,048 steps, and five seeds: 42, 137, 256, 789, and 1024.

```bash
PYTHONPATH=code python -m coevo_drl.cli train --steps 1000000 --rollout-steps 2048 --device cuda --output coevo_drl.pt
```

Hardware type, GPU count, VRAM, wall-clock duration, precision, batch size, update epochs, minibatch size, GAE coefficient, value coefficient, gradient clipping, and storage are not reported in the supplied manuscript. The implementation therefore does not claim a manuscript-matched compute footprint for these fields.

## Evaluation

The metrics module provides Harrell's concordance index, 1,000-resample percentile bootstrap intervals, Holm–Bonferroni correction for six primary comparisons, and Benjamini–Hochberg correction for exploratory comparisons. Expected manuscript targets are 0.784 ± 0.018 on SU2C-MARK and 0.728 ± 0.022 on the MSK-IMPACT NSCLC subset across five seeds. Matching these figures requires the exact authorized cohort exports and the missing supplementary parameter tables.

## Repository layout

`code/coevo_drl/model.py` contains the ODE system and phase rules. `environment.py` constructs the 15-dimensional state, five actions, episode termination, and reward. `agent.py` contains the shared 256/128 feature extractor and PPO optimization. `data.py` maps authorized molecular records to virtual-patient initial states. `metrics.py` contains survival concordance and statistical corrections. `baselines.py` contains clinical and biomarker policy comparators. `sensitivity_catalog.py` enumerates deterministic perturbation and seed combinations for parameter stability analysis.

## Numerical boundaries

ODE integration uses adaptive RK45 with relative tolerance `1e-7`, absolute tolerance `1e-9`, and a maximum integration step of 0.25 days. These numerical values are explicit implementation settings because the manuscript does not report an ODE solver or tolerance. State clipping prevents invalid negative populations after numerical integration. Patient-level cross-validation must keep each patient out of both virtual-patient generation and test evaluation for the corresponding fold.
