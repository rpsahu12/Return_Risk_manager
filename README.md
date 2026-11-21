# Return-Risk Manager

An e-commerce return-risk analysis and batch-scoring repository. It prepares an order dataset, evaluates leakage-safe features with chronological train/test splits, trains a Random Forest, explains predictions with SHAP, and exposes the trained model through Flask or FastAPI endpoints.

The project is a proof-of-concept for identifying orders that are likely to be returned before the outcome is known. It is not a persisted, production model service: the Flask and FastAPI applications retrain the final model at startup and build their scoring reference from the training portion of the included CSV.

## Repository layout

| Path | Purpose |
| --- | --- |
| `returns_sustainability_dataset.csv` | 5,000-row synthetic e-commerce order dataset used for training and evaluation |
| `data_preparation.py` | Loads and cleans data, creates leakage-safe features, encodes categoricals, and performs the chronological split |
| `model_training.py` | Tunes a Random Forest, reports test metrics and costs, and sweeps decision thresholds |
| `model_utils.py` | Stores the selected final Random Forest parameters and fits the final model |
| `model_comparison.py` | Compares Random Forest and gradient-boosting families, with and without interaction features |
| `signal_isolation.py` | Compares full, minimal, and regularized feature strategies |
| `split_diagnostics.py` | Checks whether the chronological split has a discount-distribution shift |
| `deployment_utils.py` | Builds historical customer/location references and aligns new-batch features with the training schema |
| `risk_scoring_app.py` | Flask batch-scoring API |
| `risk_scoring_app_fastapi.py` | FastAPI batch-scoring API with optional SHAP explanations |
| `shap_explainability.py` | Creates global and per-order SHAP plot images for offline analysis |
| `shap_reasons.py` | Lightweight SHAP contribution extraction for JSON API responses |
| `test.ipynb` | Exploratory notebook |

## How the model works

The training pipeline follows these steps:

1. Load `returns_sustainability_dataset.csv` and parse order dates.
2. Create `user_prior_orders`, `user_prior_return_rate`, and `is_first_order` in chronological order per customer.
3. Map `Return_Status` to `1` for `Returned` and `0` for `Not Returned`.
4. Remove outcome-only columns such as `Return_Date`, `Return_Reason`, `Days_to_Return`, `Return_Cost`, `Profit_Loss`, `CO2_Saved`, and `Waste_Avoided`.
5. Bucket uncommon locations, one-hot encode categorical features, and align the final feature matrix.
6. Split by `Order_Date`: the earliest 80% of rows is used for training and the latest 20% is held out.
7. Tune a regularized Random Forest with 3-fold cross-validation scored by ROC-AUC.
8. Fit the final model and score with an operating threshold of `0.3` instead of the generic `0.5` default.

The deployment code preserves consistency for new batches by deriving customer history from historical training-period data, using the original top-location buckets, and reindexing one-hot features to the exact training columns. It does not recompute customer history from the batch being scored.

## Setup

Windows and PowerShell are assumed.

```powershell
cd D:\AI_Risk_manager
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\Return_Risk_manager\requirements.txt
```

For the FastAPI service, install its additional runtime dependencies:

```powershell
python -m pip install fastapi "uvicorn[standard]" python-multipart
```

The model-comparison script can use XGBoost and LightGBM when installed. It automatically skips either family if the corresponding package is unavailable.

## Running the scripts

Run commands from the repository parent (`D:\AI_Risk_manager`) so the project package and local modules resolve consistently. Set `PYTHONPATH` for the current PowerShell session:

```powershell
$env:PYTHONPATH = "D:\AI_Risk_manager;D:\AI_Risk_manager\Return_Risk_manager"
cd D:\AI_Risk_manager\Return_Risk_manager
```

### Prepare data

```powershell
python data_preparation.py
```

### Train and evaluate

```powershell
python model_training.py
```

This prints the tuned parameters, held-out classification metrics, a cost table, and a threshold sweep.

### Compare model families

```powershell
python model_comparison.py
```

### Run the remaining diagnostics

```powershell
python signal_isolation.py
python split_diagnostics.py
python shap_explainability.py
```

The SHAP script writes `shap_global_summary.png` and up to three `shap_flagged_order_*.png` plots in the current working directory.

## Batch-scoring APIs

Both APIs train or load the final model in memory at startup. Data is read from `returns_sustainability_dataset.csv`, and every process startup retrains it.

### Flask

```powershell
$env:PYTHONPATH = "D:\AI_Risk_manager;D:\AI_Risk_manager\Return_Risk_manager"
cd D:\AI_Risk_manager
python .\Return_Risk_manager\risk_scoring_app.py
```

The Flask service listens on `http://localhost:7861`.

### FastAPI

```powershell
$env:PYTHONPATH = "D:\AI_Risk_manager;D:\AI_Risk_manager\Return_Risk_manager"
cd D:\AI_Risk_manager
uvicorn Return_Risk_manager.risk_scoring_app_fastapi:app --host 0.0.0.0 --port 7861
```

Interactive documentation is available at `http://localhost:7861/docs`.

## API usage

### Health check

Both implementations expose:

```text
GET /health
```

The response includes service status, the number of training orders, cutoff date, and operating threshold.

### Batch score

```text
POST /score_batch
```

Upload a raw order CSV as multipart form data in a field named `file`. An optional `threshold` query parameter overrides the default `0.3`.

PowerShell example:

```powershell
curl.exe -F "file=@new_orders.csv" "http://localhost:7861/score_batch?threshold=0.3"
```

FastAPI also accepts `explain=true|false`:

```powershell
curl.exe -F "file=@new_orders.csv" "http://localhost:7861/score_batch?threshold=0.3&explain=true"
```

### Required CSV columns

The upload must contain these columns:

- `User_ID`
- `Product_Category`
- `Product_Price`
- `Order_Quantity`
- `Discount_Applied`
- `User_Age`
- `User_Gender`
- `User_Location`
- `Payment_Method`
- `Shipping_Method`
- `CO2_Emissions`
- `Packaging_Waste`

`Order_ID` is optional in the input and is preserved in each result when present. Do not include outcome-only fields such as `Return_Status`, `Return_Date`, `Return_Reason`, `Days_to_Return`, `Return_Cost`, or `Profit_Loss`; the scoring endpoints reject or ignore them safely, but they are not required and must not be used to predict the outcome.

Example response structure:

```json
{
  "summary": {
    "total_orders": 100,
    "flagged_count": 24,
    "flag_rate": 0.24,
    "avg_predicted_risk": 0.21,
    "borderline_count": 8,
    "threshold_used": 0.3
  },
  "orders": [
    {
      "Order_ID": "ORD-123",
      "predicted_risk": 0.82,
      "flagged": true,
      "confidence": "clear",
      "top_reasons": [
        {
          "feature": "Discount_Applied",
          "contribution": 0.12,
          "order_value": "40.0"
        }
      ]
    }
  ]
}
```

The Flask implementation returns the same core fields but does not include SHAP `top_reasons`.

## Cost and interpretability assumptions

- False-negative costs use the real `Return_Cost` value from the dataset.
- False-positive cost is assumed to be `50.0` per order because the dataset has no friction, support, or intervention-cost column.
- The false-positive assumption is configurable in `model_training.py` and is not claimed to be measured cost data.
- SHAP explains model contributions relative to the selected binary return-risk class; positive contributions increase predicted risk.
- Generated risk scores are decision support, not a statement that a specific intervention prevented a return.

## Testing and validation

There is no separate automated test suite in the repository. The smallest validation steps are:

```powershell
python data_preparation.py
python model_training.py
python -c "from Return_Risk_manager.risk_scoring_app import app; print(app.url_map)"
```

For FastAPI, verify startup and OpenAPI without starting a long-lived server:

```powershell
python -c "from Return_Risk_manager.risk_scoring_app_fastapi import app; print(app.title)"
```

## License

This repository is licensed under the MIT License. See `LICENSE`.