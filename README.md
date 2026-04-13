# Trustworthy Synthetic Data Generation

Source code for the paper **"Operationalizing Trustworthy AI Practice for Synthetic Data Generation"**.

## Overview

This repository implements a four-phase requirements-driven approach that operationalises EU AI Act obligations — fairness (Article 6), security (Article 15), and explainability (Article 13) — as active training constraints for synthetic tabular data generation.

## Repository Structure

```
trustworthy_synthetic/
├── data/
│   └── preprocessing.py        Phase 2: profile-driven data preprocessing
├── models/
│   ├── profiles.py             Phase 1: regulatory profile definitions
│   ├── vae_gan.py              Phase 4: hybrid VAE-GAN architecture
│   └── losses.py               Phase 4: multi-objective loss components
├── engine/
│   └── compliance_engine.py    Phase 4: Closed-Loop Compliance Engine
├── experiments/
│   ├── run_profile_p1.py       Profile P-1: fairness experiment
│   ├── run_profile_p3.py       Profile P-2: security experiment
│   ├── run_profile_p3_exp.py   Profile P-3: explainability experiment
│   └── run_ablation.py         Cumulative ablation study
├── utils/
│   └── metrics.py              T-AI metric computation
├── run_all.py                  Single entry point for all experiments
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

Download the Insurance Claim Analysis dataset from:
https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health

Place the CSV file at `data/insurance.csv`.

For the Adult Income dataset:
https://archive.ics.uci.edu/dataset/2/adult

Place the CSV file at `data/adult.csv`.

## Running Experiments

Run all profiles across all five seeds:

```bash
python run_all.py data/insurance.csv
```

Run individual profiles:

```bash
python experiments/run_profile_p1.py data/insurance.csv
python experiments/run_profile_p3.py data/insurance.csv
python experiments/run_profile_p3_exp.py data/insurance.csv
```

Run the ablation study:

```bash
python experiments/run_ablation.py data/insurance.csv
```

## Outputs

All results are saved to `results/` as JSON files:

- `p1_result_seed{N}.json` — Profile P-1 metrics and final compliance status
- `p2_result_seed{N}.json` — Profile P-2 metrics and final compliance status
- `p3_result_seed{N}.json` — Profile P-3 metrics and final compliance status
- `p1_traceability_seed{N}.json` — Full regulatory traceability log for P-1
- `p2_traceability_seed{N}.json` — Full regulatory traceability log for P-2
- `p3_traceability_seed{N}.json` — Full regulatory traceability log for P-3
- `ablation_seed{N}.json` — Ablation study results
- `summary.json` — Aggregated means and standard deviations across all seeds

## Regulatory Profiles

| Profile | EU AI Act | Dimension | Threshold |
|---|---|---|---|
| P-1: High-Risk Medical | Article 6 | Fairness | Equalized Odds < 0.05 |
| P-2: Safety-Critical | Article 15 | Security | Attack Rate < 15% |
| P-3: Audit-Ready | Article 13 | Explainability | SHAP Stability > 0.90 |

## Reproducibility

All experiments use five fixed random seeds: 42, 123, 456, 789, 1024. Results are reported as mean ± standard deviation across seeds. Setting `torch.manual_seed` and `numpy.random.seed` at the start of each run ensures full reproducibility on the same hardware.

## Hardware

Experiments were run on a single GPU. CPU execution is supported and automatically selected when no GPU is available.
