import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
RANDOM_SEED = 42

def splitData(X , Y ):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=RANDOM_SEED, stratify=Y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=RANDOM_SEED, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test

## load data
DATA_PATH = "heart.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset shape (raw):", df.shape)
print("\nClass distribution (raw):\n", df["HeartDisease"].value_counts())

## feature / target split
TARGET_COL     = "HeartDisease"
NUMERICAL_COLS = ["Age","Oldpeak","MaxHR", "RestingBP", "Cholesterol"]
Categorical_COLS = [col for col in df.columns if col not in NUMERICAL_COLS + [TARGET_COL]]
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

# One-hot encode categorical columns before splitting
if Categorical_COLS:
    X = pd.get_dummies(X, columns=Categorical_COLS, drop_first=False)

feature_names   = X.columns.tolist()
N_TOTAL_FEATURES = len(feature_names)          
print(f"\nTotal features after one-hot encoding: {N_TOTAL_FEATURES}")
print(f"Sample features: {feature_names[:10]}")

## split data
X_train, X_val, X_test, y_train, y_val, y_test = splitData(X , y)
print(f"\nSplit sizes → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("Train class dist:\n", y_train.value_counts())

## feature preprocessing
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled   = X_val.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
X_val_scaled[NUMERICAL_COLS]   = scaler.transform(X_val[NUMERICAL_COLS])
X_test_scaled[NUMERICAL_COLS]  = scaler.transform(X_test[NUMERICAL_COLS])


def entropy(labels, weights):
    total = weights.sum()
    if total == 0:
        return 0.0
    p1 = weights[labels == 1].sum() / total
    p0 = 1.0 - p1
    if p0 == 0 or p1 == 0:
        return 0.0
    ent = 0.0
    for p in (p0, p1):
        if p > 0:
            ent -= p * np.log2(p)
    return ent


def information_gain(values, labels, weights, threshold=None):
    base_entropy = entropy(labels, weights)
    if threshold is None:
        left_mask = values == 1
    else:
        left_mask = values <= threshold
    right_mask = ~left_mask
    left_weight = weights[left_mask].sum()
    right_weight = weights[right_mask].sum()
    total_weight = weights.sum()
    if left_weight == 0 or right_weight == 0:
        return 0.0
    left_entropy = entropy(labels[left_mask], weights[left_mask])
    right_entropy = entropy(labels[right_mask], weights[right_mask])
    weighted_ent = (left_weight / total_weight) * left_entropy + (right_weight / total_weight) * right_entropy
    return base_entropy - weighted_ent


def stump_prediction(values, stump):
    values = np.asarray(values, dtype=float)
    mask = values <= stump["threshold"]
    preds = np.where(mask, stump["left_class"], stump["right_class"])
    return preds

def fit_stump(values, labels, weights):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    unique_values = np.unique(values)

    best = {
        "threshold": None,
        "left_class": 0,
        "right_class": 1,
        "ig": -1.0,
        "is_binary": False,
    }

    if set(unique_values) <= {0.0, 1.0}:
        threshold = 0.5
        left_mask = values <= threshold
        right_mask = ~left_mask
        left_class = 1 if weights[left_mask][labels[left_mask] == 1].sum() >= weights[left_mask][labels[left_mask] == 0].sum() else 0 
        right_class = 1 if weights[right_mask][labels[right_mask] == 1].sum() >= weights[right_mask][labels[right_mask] == 0].sum() else 0
        ig = information_gain(values, labels, weights, threshold)
        best.update({
            "threshold": threshold,
            "left_class": left_class,
            "right_class": right_class,
            "ig": ig,
            "is_binary": True,
        })
        return best

    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_labels = labels[sorted_idx]
    sorted_weights = weights[sorted_idx]
    thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
    thresholds = np.unique(thresholds)

    for threshold in thresholds:
        left_mask = sorted_values <= threshold
        if left_mask.all() or (~left_mask).all(): # skip if all values go to one side because IG will be zero
            continue
        left_class = 1 if sorted_weights[left_mask][sorted_labels[left_mask] == 1].sum() >= sorted_weights[left_mask][sorted_labels[left_mask] == 0].sum() else 0
        right_class = 1 if sorted_weights[~left_mask][sorted_labels[~left_mask] == 1].sum() >= sorted_weights[~left_mask][sorted_labels[~left_mask] == 0].sum() else 0
        ig = information_gain(values, labels, weights, threshold)
        if ig > best["ig"]:
            best.update({
                "threshold": threshold,
                "left_class": left_class,
                "right_class": right_class,
                "ig": ig,
            })

    if best["ig"] < 0:
        best["ig"] = 0.0
    return best


def ada_predict(X, weak_learners):
    preds = np.zeros(len(X), dtype=float)
    for learner in weak_learners:
        h = stump_prediction(X[learner["feature"]], learner)
        preds += learner["alpha"] * (2 * h - 1)  # convert to -1, 1
    return (preds > 0).astype(int)

def evaluate_model(X_test, y_test, weak_learners):
    y_pred = ada_predict(X_test, weak_learners)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return acc, f1, cm

# Train multiple AdaBoost models with resampling
n_rounds = 50
n_estimators = 10  # stumps per model
models = []
performances = []

best_acc = 0.0
no_improvement_count = 0
max_no_improvement = 5  # early stopping threshold

for round_num in range(n_rounds):
    print(f"\n--- Round {round_num + 1} ---")
    
    # Initialize weights and data
    weights = np.ones(len(X_train_scaled), dtype=float) / len(X_train_scaled)
    X_current = X_train_scaled.copy()
    y_current = y_train.copy()
    weak_learners = []
    
    for iteration in range(n_estimators):
        # Find best stump
        best_stump = None
        for col in feature_names:
            stump = fit_stump(X_current[col], y_current, weights)
            stump["feature"] = col
            if best_stump is None or stump["ig"] > best_stump["ig"]:
                best_stump = stump
        
        # Compute predictions and error
        predictions = stump_prediction(X_current[best_stump["feature"]], best_stump)
        y_array = np.asarray(y_current, dtype=int)
        misclassified = predictions != y_array
        error = weights[misclassified].sum()
        
        if error == 0 or error >= 0.5:
            print(f"Stopping at iteration {iteration + 1}: error = {error:.4f}")
            if weak_learners:
                weak_learners.append(best_stump)
            break
        
        # Compute alpha
        alpha = 0.5 * np.log((1 - error) / error)
        best_stump["alpha"] = alpha
        
        # Update weights
        weights[misclassified] = weights[misclassified] / (2 * error)
        weights[~misclassified] = weights[~misclassified] / (2 * (1 - error))
        weights /= weights.sum()
        
        weak_learners.append(best_stump)
        print(f"Iteration {iteration + 1}: selected '{best_stump['feature']}' with IG={best_stump['ig']:.4f}, error={error:.4f}, alpha={alpha:.4f}")
        
        # Resample with replacement based on weights
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(X_current), size=len(X_current), replace=True, p=weights)
        X_current = X_current.iloc[indices].reset_index(drop=True)
        y_current = y_current.iloc[indices].reset_index(drop=True)
        
        # Reset weights to uniform
        weights = np.ones(len(X_current), dtype=float) / len(X_current)
    
    # Evaluate the model
    acc, f1, cm = evaluate_model(X_test_scaled, y_test, weak_learners)
    performances.append((acc, f1, cm))
    
    print(f"Round {round_num + 1} Performance: Accuracy={acc:.4f}, F1={f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Early stopping
    if acc > best_acc:
        best_acc = acc
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= max_no_improvement:
            print(f"Early stopping at round {round_num + 1}: no improvement in {max_no_improvement} rounds")
            break
    
    models.append(weak_learners)

# Summary
print("\n=== Summary ===")
for i, (acc, f1, cm) in enumerate(performances):
    print(f"Model {i+1}: Accuracy={acc:.4f}, F1={f1:.4f}")

# Find the best model
best_idx = np.argmax([acc for acc, _, _ in performances])
best_acc, best_f1, best_cm = performances[best_idx]
print(f"\nBest Model: {best_idx+1} with Accuracy={best_acc:.4f}, F1={best_f1:.4f}")
print("Best Confusion Matrix:")
print(best_cm)

# Most confusing classes: find the highest off-diagonal value
off_diag = best_cm - np.diag(np.diag(best_cm))
max_conf = np.max(off_diag)
conf_i, conf_j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
print(f"\nMost confusing classes: Class {conf_i} misclassified as Class {conf_j} ({max_conf} times)")

# Plot confusion matrix for best model
plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Best Model (Round {best_idx+1})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



