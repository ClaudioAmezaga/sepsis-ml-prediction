# Language model-assisted label refinement for sepsis detection

Code accompanying the study **"Language model-assisted label refinement for accurate sepsis detection from electronic health records"**.

This repository shares the analysis code only. The clinical electronic health record (EHR) data, physician-adjudicated labels, and record-level language-model outputs are **not publicly available** and are **not included**. Consequently, this is a code-availability release, **not** an end-to-end reproducible dataset or an executable public demo.

## Repository contents

| File | Purpose |
| --- | --- |
| `Model_Development.ipynb` | Main research workflow: cohort preparation, model development, evaluation, sensitivity analyses, and tables/figures. |
| `functions_changes.py` | Data-processing, clinical-rule, SQL-query, feature-engineering, and evaluation helper functions imported by the notebook. |
| `requirements.txt` | Python packages used by the code (unpinned; original execution environment has not been independently recreated). |
| `.gitignore` | Prevents common local datasets, model artifacts, outputs, and credentials from being committed accidentally. |

## Analysis overview

The notebook:

1. Retrieves the septic and non-septic EHR cohorts from an access-restricted SQL Server database.
2. Incorporates a separate physician-adjudicated validation cohort and precomputed Llama-based label-refinement results.
3. Preprocesses longitudinal clinical measurements and constructs 8-, 24-, and 48-hour observation windows.
4. Trains and evaluates predictive models, including XGBoost and an additional random-forest comparison, against clinical rule-based / existing-score baselines (SIRS, SOFA, and Epic).
5. Computes discrimination, calibration, operating-point, subgroup, missingness, and feature-importance analyses, including SHAP plots and manuscript tables/figures.

**Scope:** This repository includes the downstream use of Llama-derived labels; it does **not** provide the complete Llama inference workflow or the restricted clinical records.

## Environment

The research notebook was developed with Python 3.12. The code has not been rerun against public data.

```bash
python -m venv .venv
# Activate the virtual environment for your operating system.
pip install -r requirements.txt
jupyter lab
```

SQL Server access additionally requires an installed Microsoft ODBC Driver for SQL Server and appropriate institutional permissions.

## Access-restricted inputs and configuration

The notebook requires authorized access to the original EHR database and institution-specific tables. The public version intentionally replaces the original server and cohort-schema identifiers with environment variables:

- `SEPSIS_ODBC_CONNECTION_STRING`: valid ODBC connection string for an authorized SQL Server connection.
- `SEPSIS_SEPTIC_COHORT_TABLE`: fully qualified SQL table/view identifier for the septic cohort.
- `SEPSIS_NONSEPTIC_COHORT_TABLE`: fully qualified SQL table/view identifier for the non-septic cohort.

For example, a connection string uses the general format `Driver={ODBC Driver 18 for SQL Server};Server=YOUR_SERVER;Database=YOUR_DATABASE;Encrypt=yes;...` (authentication is environment-specific). **Never commit credentials or real patient data.**

Two additional private inputs are read directly by the notebook and must be supplied through the approved research environment:

- `ValidationCohort_Final.xlsx` — physician-adjudicated encounter-level validation labels.
- `llama_results_final.jsonl` — precomputed encounter-level Llama label-refinement outputs.

Optional figure typography can be configured using `SEPSIS_FONT_DIR`, pointing to an authorized local directory of Arial font files. No fonts are distributed here.

Further database tables (e.g., medication, demographic, diagnosis, laboratory, and Epic-score records) are accessed within the notebook or helper functions. They are also restricted and are not included here. The notebook writes intermediate model-result, SHAP-cache, PDF/PNG-figure, and CSV-table files locally.

## Running the analysis

1. Obtain the necessary institutional authorization and access to the original data and SQL Server.
2. Install the dependencies, configure the three environment variables above, and place the two restricted input files where the notebook expects them (or adapt the local paths).
3. Open `Model_Development.ipynb` in Jupyter and execute cells in order. Several later cells reuse previously computed variables and intermediate artifacts.

Without these inputs, notebook execution will fail as expected. The provided source code can still be inspected to understand the implemented research methods.

## Sharing and privacy

- All notebook cell outputs and execution history have been removed before publication to avoid exposing potentially identifiable records or cohort-level research outputs.
- The SQL Server hostname, individual-specific SQL schema, and personal font-directory paths have been replaced with configuration variables.
- No raw patient-level datasets, Llama output files, saved models, or serialized intermediate data are included.
- The helper module retains SQL query structures and internal table names needed to understand the methodology. Obtain institutional approval to make the full source code public where required.

## Citation and license

If you use this code, please cite the associated manuscript by its published bibliographic details when available. No open-source license is included here; add a license only after confirming that the relevant authors and institution authorize public redistribution and reuse.
