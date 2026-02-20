# PART-4-PROJECT

# Student Suicide Risk Prediction Dashboard

An AI-powered dashboard for counselors and student welfare officers to monitor and intervene based on predictions from an XGBoost model explained via SHAP (Explainable AI).

## Features

- **Risk Prediction**: Binary classification (high/low risk) with threshold visualization
- **SHAP Explanations**: Feature contributions explained with horizontal bar charts
- **Plain Language Explanations**: Narrative summaries for non-technical users
- **Evidence-Based Recommendations**: Actionable suggestions based on risk factors
- **Risk Trajectory**: Temporal visualization showing risk over time
- **Audit Trail**: Logging for accountability and compliance

## Tech Stack

- **Backend**: Python with FastAPI, XGBoost, SHAP
- **Frontend**: React with Tailwind CSS, Recharts
- **ML**: XGBoost classifier with SHAP explainability

## Model Improvements (v2)

The model has been enhanced with:

- **Class Imbalance Handling**: Uses SMOTE (if available) or scale_pos_weight
- **Better Evaluation Metrics**: AUC-ROC, F1-Score, Precision-Recall
- **Feature Scaling**: StandardScaler for normalized inputs
- **Stratified Splitting**: Maintains class ratios in train/test
- **Hyperparameter Tuning**: Optimized for imbalanced data
- **Early Stopping**: Prevents overfitting

## Project Structure

```
XAI/
├── README.md
├── backend/
│   ├── app.py              # FastAPI application with improved ML
│   └── requirements.txt    # Python dependencies
└── frontend/
    ├── package.json
    ├── public/index.html
    ├── src/
    │   ├── App.tsx
    │   ├── index.tsx
    │   └── index.css
    ├── tailwind.config.js
    ├── postcss.config.js
    └── tsconfig.json
```

## Installation & Running

### Backend

1. Navigate to the backend directory:
```
bash
cd C:/Users/hp/Desktop/XAI/backend
```

2. Create a virtual environment (optional):
```
bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```
bash
pip install -r requirements.txt
```

4. Run the backend:
```
bash
python app.py
```

The API will be available at `http://localhost:8000`

### Frontend

1. Navigate to the frontend directory:
```
bash
cd C:/Users/hp/Desktop/XAI/frontend
```

2. Install dependencies:
```
bash
npm install
```

3. Start the development server:
```
bash
npm start
```

The dashboard will open at `http://localhost:3000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health status |
| GET | `/model_info` | Model information |
| POST | `/predict` | Get risk prediction |
| POST | `/collect_data` | Collect student data |
| POST | `/retrain` | Retrain the model |

## Example Request

```
json
POST http://localhost:8000/predict
{
  "student_id": "STU001",
  "sleep_hours": 6.5,
  "sleep_irregularity": 0.4,
  "gpa": 3.2,
  "gpa_drop": -0.25,
  "club_attendance": 2,
  "phone_hours": 8
}
```

## Model Evaluation

The model is evaluated using multiple metrics suitable for imbalanced classification:

- **Accuracy**: Overall correctness
- **AUC-ROC**: Discrimination ability
- **F1-Score**: Balance between precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

## Privacy & Ethics

- All data is anonymized
- Complies with privacy laws (GDPR, FERPA)
- For educational and prevention purposes only
- Not intended for clinical diagnosis
- Model should be trained on real historical data for production use
- Regular retraining recommended with new data

## License

For educational purposes only.
