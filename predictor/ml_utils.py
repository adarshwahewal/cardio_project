import joblib
import pandas as pd
import numpy as np

model = joblib.load("predictor/ml_model/best_xgb_pipeline.pkl")

# Human-readable labels for SHAP output
FEATURE_LABELS = {
    'age':         'Age',
    'gender':      'Gender',
    'height':      'Height',
    'weight':      'Weight',
    'ap_hi':       'Systolic BP',
    'ap_lo':       'Diastolic BP',
    'cholesterol': 'Cholesterol',
    'gluc':        'Glucose',
    'smoke':       'Smoking',
    'alco':        'Alcohol',
    'active':      'Physical Activity',
    'bmi':         'BMI',
    'age_years':   'Age (years)',
    'age_bin':     'Age Group',
}

def build_dataframe(data_dict):
    """Shared DF builder — used by both predict and SHAP"""
    df = pd.DataFrame([data_dict])
    df['age_years'] = (df['age'] / 365).round().astype(int)
    df['bmi']       = df['weight'] / ((df['height'] / 100.0) ** 2)
    df['age_bin']   = pd.cut(
        df['age_years'],
        bins=[0, 30, 45, 60, 200],
        labels=['<30', '30-45', '45-60', '60+']
    )
    df['id'] = 0
    return df


def predict_cardio(data_dict):
    df = build_dataframe(data_dict)
    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return int(prediction), float(probability)


def get_shap_explanation(data_dict):
    """
    Returns:
      contributions — list of dicts sorted by absolute impact:
        { feature, label, value, shap_value, direction, pct_impact }
      top3_reasons  — human-readable strings for UI display
    """
    try:
        import shap
    except ImportError:
        return [], []

    df = build_dataframe(data_dict)

    # Extract the trained XGBoost step from pipeline
    # Works for sklearn Pipeline with any named final step
    classifier = model.named_steps[list(model.named_steps)[-1]]

    # Transform all steps except the classifier
    preprocessor = model[:-1]
    try:
        X_transformed = preprocessor.transform(df)
    except Exception:
        # If no preprocessor steps, use raw df
        X_transformed = df

    explainer   = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    # shap_values shape: (1, n_features) for binary classification
    if isinstance(shap_values, list):
        # Older shap versions return list [class0, class1]
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Get feature names after transformation
    try:
        feat_names = preprocessor.get_feature_names_out()
    except Exception:
        feat_names = df.columns.tolist()

    # Build contribution list
    total_abs = sum(abs(v) for v in sv) or 1.0
    contributions = []
    for fname, sval in zip(feat_names, sv):
        # Strip sklearn prefixes like "remainder__ap_hi" → "ap_hi"
        clean = fname.split('__')[-1]
        contributions.append({
            'feature':    clean,
            'label':      FEATURE_LABELS.get(clean, clean.replace('_', ' ').title()),
            'shap_value': round(float(sval), 5),
            'direction':  'increases' if sval > 0 else 'decreases',
            'pct_impact': round(abs(float(sval)) / total_abs * 100, 1),
        })

    # Sort by absolute impact
    contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    top5 = contributions[:5]

    # Top 3 human-readable reasons
    top3_reasons = []
    for c in top5[:3]:
        direction_text = 'raised' if c['direction'] == 'increases' else 'reduced'
        top3_reasons.append(
            f"{c['label']} {direction_text} risk by {c['pct_impact']}%"
        )

    return top5, top3_reasons