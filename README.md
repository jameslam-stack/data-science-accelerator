# Data Science Accelerator

A collection of lesson plans, notebooks, and resources from the **Data Science Accelerator** programme.

This repository contains the full workshop content — including lesson plans, learner and teacher notebooks, code-along materials, datasets, and supporting reports — organised by module.

## Repository structure

| Folder | Description |
| --- | --- |
| `Launch and foundations/` | Module 1 — Python foundations, launch workshops, slides, and resources |
| `AutoML/` | Module 7 — Rapid EDA and AutoML (learner, teacher, code-along, capstone) |
| `Time Series/` | Time series forecasting modules and datasets |
| `Stats/` | Statistics, feature selection, and EDA reports |
| `RnC/` | Research & Communication module content |
| `EDA/` | Exploratory Data Analysis notebooks and outputs |
| `Python Data Processing/` | Python data processing exercises and notebooks |
| `Intro Machine learning/` | Introductory machine learning material |
| `capstone/` | Capstone project work |
| `Data science projects/` | Week 5 university project demo and examples |
| `data/` | Shared datasets |
| `reports/` | Generated reports |
| `Users/` | User-specific or learner materials |

## Top-level files

| File | Description |
| --- | --- |
| `data_cleaning_pipeline.py` | Data cleaning pipeline implementation |
| `generate_skills.py` | Skill generation utility |
| `geometric_series_sum.py` | Example Python script |
| `gov_service_requests.csv` | Sample government service requests dataset |
| `messy_sample.csv` | Messy data sample for cleaning exercises |
| `uk_defra_air_quality.csv` | UK DEFRA air quality dataset |

## Modules

- **Launch and foundations** — Python basics, environment setup, first workshops
- **AutoML** — Rapid EDA, feature engineering, model building (learner + teacher + code-along)
- **Time Series** — Forecasting with notebooks and datasets
- **Stats** — Statistics, feature selection, EDA
- **RnC** — Research & Communication
- **EDA** — Exploratory Data Analysis
- **Python Data Processing** — Data processing fundamentals
- **Intro Machine Learning** — Intro ML concepts
- **Capstone** — Capstone project

## Getting started

Each module folder typically contains:
- `M*_Lesson plan.md` / `.docx` — lesson plan
- `M*_learner.ipynb` / `M*_learner_completed.ipynb` — learner notebooks
- `M*_teacher.ipynb` — teacher version
- `M*_code_along.ipynb` — code-along materials
- Datasets (`.csv`) and generated reports (`.html`)

Run any notebook with Jupyter:

```bash
jupyter notebook AutoML/M7.1_Rapid_EDA_learner.ipynb
```

## Notes

- Checkpoint files (`.ipynb_checkpoints/`) are included for version history.
- Some folders contain model artefacts (`.pkl`, `.db`) and HTML reports.
- Teacher resources and SME peer review notes are marked for internal use.

## Contributing

This is a workshop resource repository. Add or update materials following the existing module naming conventions.