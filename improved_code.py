# -*- coding: utf-8 -*-
"""improved_code.py

Improved version of the network intrusion detection model
with fixes for low accuracy issues and better class imbalance handling.
"""

import sklearn as sk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import os
import sys

# Set parameters - ADJUSTED FOR BETTER PERFORMANCE
OUTLIER_THRESHOLD = 0.1
FEATURE_THRESHOLD = 0.01  # CHANGED: Less aggressive (was 0.1)
BIN_THRESHOLD = 0.02      # CHANGED: Less aggressive (was 0.1)
USE_SMOTE = True          # NEW: Enable SMOTE for class imbalance
USE_LOG_TRANSFORM = False # NEW: Make log transform optional

np.random.seed(42)

# Load data
if os.path.isdir(r'C:\Users\User\Desktop\Data-mining-group-project'):
    os.chdir(r'C:\Users\User\Desktop\Data-mining-group-project')
elif os.path.isdir(r'/content/drive/MyDrive/data mining'):
    os.chdir(r'/content/drive/MyDrive/data mining')
    
try:
    train = pd.read_csv(open('UNSW_NB15_training-set.csv', encoding='utf-8'))
    test = pd.read_csv(open('UNSW_NB15_testing-set.csv', encoding='utf-8'))
except FileNotFoundError as e:
    print(f"error: {e}\nTry changing the training data directory in 'os.chdir'")

print(train.shape, test.shape)

# Prepare data
train = train.sample(frac=1)
test = test.sample(frac=1)
x_train, y_train = train.iloc[:, :-2], train.iloc[:, -2:]
x_test, y_test = test.iloc[:, :-2], test.iloc[:, -2:]

# NEW FUNCTION: Analyze class imbalance
def analyze_class_imbalance(y_train, y_test):
    """Comprehensive class imbalance analysis"""
    print("="*60)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*60)
    
    # Training set analysis
    train_counts = y_train['label'].value_counts()
    train_pct = y_train['label'].value_counts(normalize=True) * 100
    
    print("\nTraining Set:")
    for label, count in train_counts.items():
        pct = train_pct[label]
        print(f"  Label {label}: {count:,} samples ({pct:.2f}%)")
    
    # Test set analysis
    test_counts = y_test['label'].value_counts()
    test_pct = y_test['label'].value_counts(normalize=True) * 100
    
    print("\nTest Set:")
    for label, count in test_counts.items():
        pct = test_pct[label]
        print(f"  Label {label}: {count:,} samples ({pct:.2f}%)")
    
    # Imbalance ratio
    majority_class = train_counts.max()
    minority_class = train_counts.min()
    imbalance_ratio = majority_class / minority_class
    
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 10:
        print("⚠️  SEVERE IMBALANCE DETECTED - SMOTE highly recommended")
    elif imbalance_ratio > 3:
        print("⚠️  Moderate imbalance - Consider using SMOTE or class_weight")
    else:
        print("✓ Relatively balanced dataset")
    
    print("="*60)
    return imbalance_ratio

# Run class imbalance analysis
imbalance_ratio = analyze_class_imbalance(y_train, y_test)

# Existing helper functions
def show_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (lower_bound, upper_bound)

def replace_outliers_iqr(df, cols):
    for col in cols:
        if col in df.columns:
            lower_bound, upper_bound = show_outliers_iqr(df[col])
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def remove_outliers_iqr(df, cols):
    overall_mask = pd.Series(True, index=df.index)
    for col in cols:
        if col in df.columns:
            lower_bound, upper_bound = show_outliers_iqr(df[col])
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            overall_mask = overall_mask & col_mask
    df = df[overall_mask]
    return df

def risky_show_shape(*arg):
    if len(arg) == 2:
        print("train/test")
        print(arg[0].shape, arg[1].shape)
    else:
        print("x_train/x_test/y_train/y_test")
        for df in arg:
            print(df.shape)

def plot_histograms(df, numerical_cols, frac=0.1, title='graph'):
    df = df.copy().sample(frac=frac)
    columns_to_plot = [col for col in numerical_cols if col in df.columns]
    num_plots = len(columns_to_plot)
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    plt.figure(figsize=(20, num_rows * 5))
    
    for i, col in enumerate(columns_to_plot):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    for j in range(i + 1, num_rows * num_cols):
        plt.subplot(num_rows, num_cols, j + 1).set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Custom Transformers
class Plot(BaseEstimator, TransformerMixin):
    def __init__(self, frac=0.1, title='graph'):
        self.frac = frac
        self.title = title

    def fit(self, X, y=None):
        print(f"Plot transformer - Shape: {X.shape}")
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            temp_df = pd.DataFrame(X)
            cols_to_plot = list(range(X.shape[1]))
        elif isinstance(X, pd.DataFrame):
            temp_df = X
            cols_to_plot = X.columns.tolist()
        else:
            raise TypeError("Input must be a pandas DataFrame or a numpy array.")
        
        plot_histograms(temp_df, cols_to_plot, self.frac, self.title)
        return X

class Add_feature(BaseEstimator, TransformerMixin):
    def __init__(self, feature_1, feature_2):
        self.feature_1 = feature_1
        self.feature_2 = feature_2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['new_feature'] = X[self.feature_1] + X[self.feature_2]
        return X

class Ratio_feature(BaseEstimator, TransformerMixin):
    def __init__(self, feature_1, feature_2):
        self.feature_1 = feature_1
        self.feature_2 = feature_2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Avoid division by zero
        denominator = X[self.feature_2].replace(0, 1)
        X['new_feature'] = X[self.feature_1] / denominator
        return X

class capping(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = replace_outliers_iqr(X, self.cols)
        return output

class Threshold_binning(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.02, other_label="Others"):
        self.threshold = threshold
        self.other_label = other_label

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.keep_ = {}
        
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.keep_[col] = freq[freq > self.threshold].index
        
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        
        for col in X.columns:
            keep = self.keep_.get(col, [])
            X[col] = X[col].where(X[col].isin(keep), self.other_label)
        return X

class Pearson_feature_selection(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def fit(self, X, y):
        X_df = pd.DataFrame(X).reset_index(drop=True) if not isinstance(X, pd.DataFrame) else X.reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True) if not isinstance(y, pd.Series) else y.reset_index(drop=True)
        
        correlations = X_df.corrwith(y_series).abs().fillna(0)
        
        self.keep_ = correlations > self.threshold
        self.n_features_in_ = X_df.shape[1]
        self.n_features_out_ = self.keep_.sum()
        
        print(f'Pearson selection: {self.n_features_in_} -> {self.n_features_out_} features (threshold={self.threshold})')
        print(f'Dropped {self.n_features_in_ - self.n_features_out_} features')
        
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.keep_]
        else:
            return X[:, self.keep_.values]

class SafeLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, replacement_value=1e-10):
        self.replacement_value = replacement_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        if hasattr(X, 'iloc'):
            X_transformed = X.replace(0, self.replacement_value)
            return np.log1p(X_transformed)
        else:
            X_transformed[X == 0] = self.replacement_value
            return np.log1p(X_transformed)

# Identify column types
numerical_cols = x_train.select_dtypes(include=np.number).columns
non_numerical_cols = x_train.select_dtypes(exclude=np.number).columns

print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"Non-numerical columns: {len(non_numerical_cols)}")

# IMPROVED: Keep original features AND add new combined features
from sklearn.preprocessing import OneHotEncoder

column_modifier_improved = Pipeline([
    ("ttl_sum", Add_feature('sttl', 'ct_state_ttl')),
    # DON'T drop original features - keep them!
])

# Define the numerical pipeline - MADE LESS AGGRESSIVE
numerical_pipeline = Pipeline([
    ("capping", capping(numerical_cols)),
    ("column_mod", column_modifier_improved),
])

# Only add log transform if enabled
if USE_LOG_TRANSFORM:
    numerical_pipeline.steps.append(("logrithmic_transform", SafeLogTransformer()))
    print("✓ Log transformation enabled")
else:
    print("✗ Log transformation disabled (recommended)")

# Define the categorical pipeline with less aggressive binning
categorical_pipeline = Pipeline([
    ("bin_rare", Threshold_binning(threshold=BIN_THRESHOLD)),
    ("one_hot_encoding", OneHotEncoder(handle_unknown='ignore'))
])

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, non_numerical_cols)
    ],
    remainder='passthrough'
)

# Main preprocessing pipeline with optional feature selection
preprocessing_steps = [("feature_preprocessing", preprocessor)]

if FEATURE_THRESHOLD > 0:
    preprocessing_steps.append(("pearson_selection", Pearson_feature_selection(threshold=FEATURE_THRESHOLD)))
    print(f"✓ Pearson feature selection enabled (threshold={FEATURE_THRESHOLD})")
else:
    print("✗ Feature selection disabled - using all features")

preprocessing = Pipeline(preprocessing_steps)

# Run pipeline
print("\n" + "="*60)
print("PREPROCESSING DATA")
print("="*60)
encoded_x_train = preprocessing.fit_transform(x_train, y_train['label'])
encoded_x_test = preprocessing.transform(x_test)

print(f"\nFinal feature count: {encoded_x_train.shape[1]}")
print(f"Training samples: {encoded_x_train.shape[0]}")
print(f"Test samples: {encoded_x_test.shape[0]}")

# NEW: Apply SMOTE if enabled
if USE_SMOTE:
    try:
        from imblearn.over_sampling import SMOTE
        print("\n" + "="*60)
        print("APPLYING SMOTE")
        print("="*60)
        print(f"Before SMOTE: {encoded_x_train.shape}")
        print(f"Class distribution before: {pd.Series(y_train['label']).value_counts().to_dict()}")
        
        smote = SMOTE(random_state=42)
        encoded_x_train_balanced, y_train_balanced = smote.fit_resample(encoded_x_train, y_train['label'])
        
        print(f"\nAfter SMOTE: {encoded_x_train_balanced.shape}")
        print(f"Class distribution after: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        # Use balanced data
        encoded_x_train = encoded_x_train_balanced
        y_train_label = y_train_balanced
        print("✓ SMOTE applied successfully")
    except ImportError:
        print("⚠️  SMOTE requested but imbalanced-learn not installed")
        print("Install with: pip install imbalanced-learn")
        y_train_label = y_train['label']
else:
    print("\n✗ SMOTE disabled - using original class distribution")
    y_train_label = y_train['label']

# Visualize label distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=y_train_label if USE_SMOTE else y_train['label'])
plt.title('Distribution of Labels in Training Set' + (' (After SMOTE)' if USE_SMOTE else ''))
plt.xlabel('Label')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x=y_test['label'])
plt.title('Distribution of Labels in Test Set')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Model training with GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
hyperparameters = {}  

# Random Forest - IMPROVED hyperparameters
param_grid_rf = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'n_estimators': [100, 200],
    'bootstrap': [True]
}

print("\nTraining Random Forest...")
# Use class_weight only if SMOTE is not used
class_weight = None if USE_SMOTE else 'balanced'
grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight=class_weight),
    param_grid=param_grid_rf,
    cv=cv_strategy,
    verbose=1,
    n_jobs=-1
)
grid_search_rf.fit(encoded_x_train, y_train_label)
hyperparameters['Random_Forest'] = grid_search_rf.best_estimator_
print(f"Best params: {grid_search_rf.best_params_}")

# Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [200, 500]
}

print("\nTraining Logistic Regression...")
grid_search_lr = GridSearchCV(
    LogisticRegression(random_state=42, class_weight=class_weight),
    param_grid=param_grid_lr,
    cv=cv_strategy,
    verbose=1,
    n_jobs=-1
)
grid_search_lr.fit(encoded_x_train, y_train_label)
hyperparameters['Logistic_Regression'] = grid_search_lr.best_estimator_
print(f"Best params: {grid_search_lr.best_params_}")

# XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

print("\nTraining XGBoost...")
# XGBoost uses scale_pos_weight instead of class_weight
if USE_SMOTE:
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
else:
    scale_pos_weight = imbalance_ratio
    xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss', random_state=42)

grid_search_xgb = GridSearchCV(
    xgb_model,
    param_grid=param_grid_xgb,
    cv=cv_strategy,
    verbose=1,
    n_jobs=-1
)
grid_search_xgb.fit(encoded_x_train, y_train_label)
hyperparameters['XGBoost'] = grid_search_xgb.best_estimator_
print(f"Best params: {grid_search_xgb.best_params_}")

# IMPROVED EVALUATION with more metrics
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

results = []

for model_name, model_estimator in hyperparameters.items():
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    y_pred = model_estimator.predict(encoded_x_test)
    
    # Check prediction distribution
    pred_dist = pd.Series(y_pred).value_counts()
    print(f"\nPrediction distribution:")
    for label, count in pred_dist.items():
        pct = (count / len(y_pred)) * 100
        print(f"  Label {label}: {count:,} ({pct:.2f}%)")
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test['label'], y_pred)
    balanced_acc = balanced_accuracy_score(y_test['label'], y_pred)
    precision = precision_score(y_test['label'], y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test['label'], y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test['label'], y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    print(f"\nClassification Report:")
    print(classification_report(y_test['label'], y_pred, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test['label'], y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Balanced_Accuracy': balanced_acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Display results
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("="*60)
print(results_df.to_string(index=False))

# Summary of improvements
print("\n" + "="*60)
print("IMPROVEMENTS IMPLEMENTED")
print("="*60)
print(f"1. Feature selection threshold: 0.1 → {FEATURE_THRESHOLD} (less aggressive)")
print(f"2. Binning threshold: 0.1 → {BIN_THRESHOLD} (less aggressive)")
print(f"3. SMOTE for class imbalance: {'ENABLED' if USE_SMOTE else 'DISABLED'}")
print(f"4. Log transformation: {'ENABLED' if USE_LOG_TRANSFORM else 'DISABLED'}")
print(f"5. Kept original features when creating new ones")
print(f"6. Added balanced_accuracy metric (better for imbalanced data)")
print(f"7. Added per-class precision/recall in classification report")
print(f"8. Added prediction distribution analysis")
