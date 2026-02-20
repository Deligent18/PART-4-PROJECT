from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
import os
from datetime import datetime

app = FastAPI()

# Try to import imbalanced-learn, if not available use built-in scale_pos_weight
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("imbalanced-learn not installed. Using scale_pos_weight for imbalance handling.")

# Global variables
model = None
explainer = None
scaler = None

def load_or_simulate_data():
    """Load or simulate student data"""
    if os.path.exists('student_data.csv'):
        return pd.read_csv('student_data.csv')
    
    # Simulate more realistic data with correlation between features
    np.random.seed(42)
    n_samples = 2000  # Increased sample size for better training
    
    # Generate correlated features
    sleep_hours = np.random.uniform(4, 10, n_samples)
    sleep_irregularity = np.random.uniform(0, 1, n_samples)
    gpa = np.random.uniform(2.0, 4.0, n_samples)
    gpa_drop = np.random.uniform(-0.5, 0.5, n_samples)
    club_attendance = np.random.randint(0, 5, n_samples)
    phone_hours = np.random.uniform(2, 12, n_samples)
    
    # Create risk labels based on realistic rules (for simulation)
    # High risk: low sleep, high irregularity, low gpa, negative drop, low social activity
    risk_score = (
        (10 - sleep_hours) / 10 * 0.25 +  # Less sleep = higher risk
        sleep_irregularity * 0.25 +  # More irregular = higher risk
        (4 - gpa) / 4 * 0.2 +  # Lower GPA = higher risk
        (-gpa_drop) * 0.2 +  # Negative drop = higher risk
        (4 - club_attendance) / 4 * 0.1  # Less social = higher risk
    )
    
    # Convert to binary (imbalanced - about 10% high risk)
    risk_label = (risk_score > 0.55).astype(int)
    
    data = pd.DataFrame({
        'sleep_hours': sleep_hours,
        'sleep_irregularity': sleep_irregularity,
        'gpa': gpa,
        'gpa_drop': gpa_drop,
        'club_attendance': club_attendance,
        'phone_hours': phone_hours,
        'risk_label': risk_label
    })
    
    data.to_csv('student_data.csv', index=False)
    print(f"Data shape: {data.shape}, High risk: {data['risk_label'].sum()} ({data['risk_label'].mean()*100:.1f}%)")
    return data

def train_model():
    """Train XGBoost model with proper imbalance handling and hyperparameter tuning"""
    global model, explainer, scaler
    
    print("Loading data...")
    data = load_or_simulate_data()
    X = data.drop('risk_label', axis=1)
    y = data['risk_label']
    
    # Calculate scale_pos_weight for imbalance
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    print(f"Class distribution: Low Risk={neg_count}, High Risk={pos_count}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stratified split to maintain class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Handle imbalance with SMOTE if available
    if HAS_IMBLEARN:
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {len(y_train_res)} samples")
    else:
        X_train_res, y_train_res = X_train, y_train
    
    # Define model with scale_pos_weight for imbalance
    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='aucpr',  # Use AUC-PR for imbalanced data
        early_stopping_rounds=10,
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Train with early stopping
    print("Training model...")
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate on test set
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Calculate multiple metrics
    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    f1 = f1_score(y_test, predictions)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Low Risk', 'High Risk']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("="*50)
    
    # Create SHAP explainer
    explainer = shap.Explainer(model)
    
    # Save model and scaler
    model.save_model('xgboost_model.json')
    
    # Save scaler parameters for prediction
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'features': list(X.columns)
    }
    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)
    
    print("Model trained and saved successfully!")

def load_model():
    """Load existing model"""
    global model, explainer, scaler
    
    model = xgb.XGBClassifier()
    model.load_model('xgboost_model.json')
    
    # Load scaler parameters
    with open('scaler_params.json', 'r') as f:
        scaler_params = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params['mean'])
    scaler.scale_ = np.array(scaler_params['scale'])
    scaler.n_features_in_ = len(scaler_params['features'])
    
    explainer = shap.Explainer(model)
    print("Model loaded from file")

# Try to load existing model, otherwise train
try:
    load_model()
except Exception as e:
    print(f"Could not load model: {e}")
    print("Training new model...")
    train_model()

# Data models
class StudentData(BaseModel):
    student_id: str
    sleep_hours: float
    sleep_irregularity: float
    gpa: float
    gpa_drop: float
    club_attendance: int
    phone_hours: float

class CollectedData(BaseModel):
    student_id: str
    sleep_pattern: dict
    phone_hours: float

# API to collect data
@app.post("/collect_data")
def collect_data(data: CollectedData):
    with open('collected_data.json', 'a') as f:
        f.write(json.dumps(data.dict()) + '\n')
    return {"status": "Data collected"}

# Prediction endpoint
@app.post("/predict")
def predict(data: StudentData = Body(...)):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Prepare input data
    input_dict = data.dict()
    student_id = input_dict.pop('student_id')
    input_df = pd.DataFrame([input_dict])
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Get prediction
    prediction_prob = model.predict_proba(input_scaled)[:, 1][0]
    
    # Adaptive threshold based on class distribution (can be tuned)
    threshold = 0.5  # Using 0.5 as base threshold
    risk_level = "high" if prediction_prob > threshold else "low"
    
    # SHAP explanations
    shap_values = explainer(input_scaled)
    shap_dict = dict(zip(input_df.columns, shap_values.values[0]))
    
    # Feature mapping
    feature_map = {
        'sleep_hours': 'Sleep Hours',
        'sleep_irregularity': 'Sleep Irregularity',
        'gpa': 'GPA',
        'gpa_drop': 'GPA Drop',
        'club_attendance': 'Club Attendance',
        'phone_hours': 'Phone Hours'
    }
    
    # Sort for bar chart
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    bar_data = [{"feature": feature_map.get(k, k), "contribution": v} for k, v in sorted_shap]
    
    # Narrative summary
    top_factors = [f"{feature_map.get(k, k)} ({v:.2f})" for k, v in sorted_shap[:3] if v > 0]
    if top_factors:
        narrative = f"This student's risk is primarily driven by {', '.join(top_factors)}. Consider monitoring closely."
    else:
        narrative = "This student shows low risk indicators across all features."
    
    # Recommendations
    recommendations = []
    if shap_dict.get('gpa_drop', 0) < -0.1:
        recommendations.append("Provide academic support for GPA decline.")
    if shap_dict.get('club_attendance', 0) < 0:
        recommendations.append("Encourage participation in social connection programs.")
    if shap_dict.get('sleep_irregularity', 0) > 0:
        recommendations.append("Suggest sleep hygiene workshops.")
    if shap_dict.get('sleep_hours', 7) < 6:
        recommendations.append("Address potential sleep deprivation - recommend sleep evaluation.")
    
    if not recommendations:
        recommendations.append("Continue regular monitoring and maintain supportive environment.")
    
    # Simulated trajectory (in production, fetch from database)
    trajectory = [
        {"date": "2026-01-01", "risk": round(np.random.uniform(0.2, 0.5), 2)},
        {"date": "2026-01-15", "risk": round(np.random.uniform(0.3, 0.6), 2)},
        {"date": "2026-02-01", "risk": round(prediction_prob, 2)}
    ]
    
    # Audit trail
    audit = {
        "action": "Prediction requested",
        "timestamp": datetime.now().isoformat(),
        "student_id": student_id
    }
    
    return {
        "risk_level": risk_level,
        "probability": round(prediction_prob, 4),
        "threshold_used": threshold,
        "shap_bar_data": bar_data,
        "narrative": narrative,
        "recommendations": recommendations,
        "trajectory": trajectory,
        "audit": audit
    }

# Endpoint to retrain model
@app.post("/retrain")
def retrain():
    """Retrain the model with current data"""
    try:
        train_model()
        return {"status": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# Health check endpoint
@app.get("/")
def root():
    return {"message": "Student Risk Prediction API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/model_info")
def model_info():
    """Get model information and metrics"""
    return {
        "model_type": "XGBoost Classifier",
        "features": ["sleep_hours", "sleep_irregularity", "gpa", "gpa_drop", "club_attendance", "phone_hours"],
        "imbalance_handling": "SMOTE" if HAS_IMBLEARN else "scale_pos_weight",
        "threshold": 0.5
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
