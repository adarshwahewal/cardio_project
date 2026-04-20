# =========================
# 📦 Imports
# =========================
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# =========================
# ⚙️ Config
# =========================
RANDOM_STATE = 42
DATA_PATH = "/Users/adarshwahewal/Desktop/mirza 12032026/cardio_train_fixed.xlsx"

# =========================
# 📊 Load Data (FIXED)
# =========================
df = pd.read_excel(DATA_PATH)

# =========================
# 🧠 Feature Engineering
# =========================
df['age_years'] = (df['age'] / 365).round().astype(int)
df['bmi'] = df['weight'] / ((df['height']/100.0)**2)
df['age_bin'] = pd.cut(
    df['age_years'],
    bins=[0,30,45,60,200],
    labels=['<30','30-45','45-60','60+']
)

# =========================
# 🎯 Split
# =========================
TARGET = "cardio"
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

# =========================
# 🔧 Preprocessing
# =========================
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# =========================
# 🚀 Model (XGBoost)
# =========================
xgb_pipe = Pipeline([
    ('preproc', preprocessor),
    ('clf', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

param_dist = {
    'clf__n_estimators': [200, 400],
    'clf__learning_rate': [0.01, 0.05],
    'clf__max_depth': [3, 5],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0]
}

# =========================
# 🔀 Train
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

rs = RandomizedSearchCV(
    xgb_pipe,
    param_dist,
    n_iter=5,
    scoring='roc_auc',
    cv=skf,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

rs.fit(X_train, y_train)

best_model = rs.best_estimator_

print("✅ Training Done")
print("Best Score:", rs.best_score_)

# =========================
# 💾 Save Model
# =========================
joblib.dump(best_model, "best_xgb_pipeline.pkl")

print("✅ Model Saved: best_xgb_pipeline.pkl")