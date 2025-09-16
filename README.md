# Dynamic Sepsis Prediction with Machine Learning
This repository contains the codebase for our research on machine learning approaches for dynamic sepsis prediction.
The project is currently under construction. The objective is to provide the full code architecture, preprocessing functions, model training scripts, and evaluation pipelines used in the study.

⚠️ Important:

No real patient data will be published here.

Only synthetic or placeholder data may be included for demonstration purposes.

sepsis-ml-prediction/
│
├── README.md                # Project description and instructions
├── requirements.txt          # Dependencies
├── LICENSE                   # License (e.g. MIT, Apache 2.0)
│
├── data_preprocessing/       # Functions for data preparation
│   ├── exclusion_criteria.py 
│   ├── imputation.py         
│   └── sliding_window.py     
│
├── models/
│   ├── train_xgboost.py      
│   ├── evaluate_model.py     
│   └── shap_analysis.py      
│
├── utils/
│   ├── cohort_selection.py   
│   └── metrics.py            
│
└── examples/
    └── synthetic_demo.ipynb  # Notebook with synthetic data demo

