# Operationalizing Trustworthy AI for Synthetic Data Generation

Official implementation of "Operationalizing Trustworthy AI Practice for Synthetic Data Generation" submitted to RCIS 2025.

## Overview

This repository contains the complete implementation of our requirements-driven framework that translates EU AI Act regulatory obligations into concrete model parameters through:

1. **Translation Mechanism**: Mathematically derives loss function weights from regulatory profiles
2. **Closed-Loop Compliance Engine**: Autonomously monitors and adjusts weights during training
3. **Hybrid VAE-GAN Architecture**: Generates high-fidelity synthetic data while enforcing T-AI constraints

## Key Features

- ✅ Three regulatory profiles (P-1: Fairness, P-2: Security, P-3: Explainability)
- ✅ Automated compliance verification with formal traceability
- ✅ 63-74% reduction in development time vs. manual tuning
- ✅ Complete audit certificate generation

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (optional, for GPU acceleration)

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/trustworthy-ai-synthetic-data.git
cd trustworthy-ai-synthetic-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Running Profile P-1 (Fairness - High-Risk Medical)
```bash
python experiments/run_profile_p1.py --dataset data/insurance_claims.csv --epochs 200 --seed 42
```

### Running Profile P-2 (Security - Safety-Critical)
```bash
python experiments/run_profile_p2.py --dataset data/insurance_claims.csv --epochs 200 --seed 42
```

### Running Profile P-3 (Explainability - Audit-Ready)
```bash
python experiments/run_profile_p3.py --dataset data/insurance_claims.csv --epochs 200 --seed 42
```

### Running Ablation Study
```bash
python experiments/run_ablation.py --dataset data/insurance_claims.csv --seeds 5
```

## Dataset

The framework is validated on the Insurance Claim Analysis Dataset:
- 1,341 patient records
- 11 features (demographics + health metrics)
- Target: insurance charges

Download from: https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health

Place the CSV file in `data/insurance_claims.csv`

## Project Structure
```
src/
├── data/           # Data loading and preprocessing
├── models/         # VAE, GAN, and hybrid architectures
├── translation/    # Translation Mechanism (Eq. 1)
├── compliance/     # Closed-Loop Engine (Algorithm 1)
├── attacks/        # AutoZOOM adversarial attack
└── utils/          # Logging, visualization, helpers
```

## Configuration

Edit `config/profiles.yaml` to customize regulatory profiles:
```yaml
profile_p1:
  name: "High-Risk Medical"
  eu_article: "Article 6"
  priorities:
    fairness: 0.6
    security: 0.3
    explainability: 0.1
  thresholds:
    equalized_odds: 0.05
    demographic_parity: 0.05
```

## Reproducing Paper Results

### Table 2: Translation Mechanism Effectiveness
```bash
python experiments/run_profile_p1.py --compare-initializations
```

### Table 3: Fairness Results (Profile P-1)
```bash
python experiments/run_profile_p1.py --compare-baselines
```

### Table 4: Security Results (Profile P-2)
```bash
python experiments/run_profile_p2.py --adversarial-attack
```

### Table 5: Explainability Results (Profile P-3)
```bash
python experiments/run_profile_p3.py --shap-analysis
```



## Citation

If you use this code in your research, please cite:
```bibtex
@inproceedings{sardar2025operationalizing,
  title={Operationalizing Trustworthy AI Practice for Synthetic Data Generation},
  author={Sardar, Bilal and Islam, Shareeful and Papastergiou, Spyridon},
  booktitle={Proceedings of the Research Challenges in Information Science (RCIS)},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details

## Contact

- Bilal Sardar: bilal.sardar@aru.ac.uk
- Shareeful Islam: shareeful.islam@aru.ac.uk
- Spyridon Papastergiou: spyros.papastergiou@maggioli.gr

## Acknowledgments

This work was supported by the European Union’s Horizon Europe Programme through the projects CUSTODES, CONSENTIS, and CyberSecDome.
