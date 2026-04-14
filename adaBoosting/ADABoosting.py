import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
RANDOM_SEED = 42

def splitData(X, Y):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=RANDOM_SEED, stratify=Y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=RANDOM_SEED, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

## load data
df = pd.read_csv("C:\\Users\\Mega Store\\Desktop\\8th semster\\pattern\\Decision-Tree-Lab\\adaBoosting\\heart.csv")
print("Dataset shape (raw):", df.shape)
print("\nClass distribution (raw):\n", df["HeartDisease"].value_counts())

## feature / target split
TARGET_COL     = "HeartDisease"
NUMERICAL_COLS = ["Age", "Oldpeak", "MaxHR", "RestingBP", "Cholesterol"]
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
X_train, X_val, X_test, y_train, y_val, y_test = splitData(X, y)
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
    left_weight  = weights[left_mask].sum()
    right_weight = weights[right_mask].sum()
    total_weight = weights.sum()
    if left_weight == 0 or right_weight == 0:
        return 0.0
    left_entropy  = entropy(labels[left_mask],  weights[left_mask])
    right_entropy = entropy(labels[right_mask], weights[right_mask])
    weighted_ent  = (left_weight / total_weight) * left_entropy + \
                    (right_weight / total_weight) * right_entropy
    return base_entropy - weighted_ent


def stump_prediction(values, stump):
    values = np.asarray(values, dtype=float)
    mask = values <= stump["threshold"]
    return np.where(mask, stump["left_class"], stump["right_class"])

def fit_stump(values, labels, weights):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    unique_values = np.unique(values)

    best = {
        "threshold":   None,
        "left_class":  0,
        "right_class": 1,
        "ig":          -1.0,
        "is_binary":   False,
    }

    if set(unique_values) <= {0.0, 1.0}:
        threshold = 0.5
        left_mask = values <= threshold
        right_mask = ~left_mask
        left_class = 1 if weights[left_mask][labels[left_mask] == 1].sum() >= \
                           weights[left_mask][labels[left_mask] == 0].sum() else 0
        right_class = 1 if weights[right_mask][labels[right_mask] == 1].sum() >= \
                           weights[right_mask][labels[right_mask] == 0].sum() else 0
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
    thresholds = np.unique((sorted_values[:-1] + sorted_values[1:]) / 2)

    for threshold in thresholds:
        left_mask = sorted_values <= threshold
        if left_mask.all() or (~left_mask).all(): # skip if all values go to one side because IG will be zero
            continue
        left_class = 1 if sorted_weights[left_mask][sorted_labels[left_mask] == 1].sum() >= \
                           sorted_weights[left_mask][sorted_labels[left_mask] == 0].sum() else 0
        right_class = 1 if sorted_weights[~left_mask][sorted_labels[~left_mask] == 1].sum() >= \
                           sorted_weights[~left_mask][sorted_labels[~left_mask] == 0].sum() else 0
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
        preds += learner["alpha"] * (2 * h - 1)  # map {0,1} → {-1,+1}
    return (preds > 0).astype(int)

def evaluate_model(X_data, y_data, weak_learners):
    y_pred = ada_predict(X_data, weak_learners)
    acc = accuracy_score(y_data, y_pred)
    f1  = f1_score(y_data, y_pred, average='weighted')
    cm  = confusion_matrix(y_data, y_pred)
    return acc, f1, cm

n_estimators        = 50   # maximum number of boosting rounds 
early_stop_patience = 5    # stop if val accuracy doesn't improve for this many rounds
weights = np.ones(len(X_train_scaled), dtype=float) / len(X_train_scaled)
best_val_acc       = 0.0
no_improve_count   = 0
weak_learners      = []      # all accepted stumps so far
best_weak_learners = []      # snapshot of the best ensemble (chosen on val set)

y_train_arr = np.asarray(y_train, dtype=int)

for iteration in range(n_estimators):
    # Find the best stump across all features
    best_stump = None
    for col in feature_names:
        stump = fit_stump(X_train_scaled[col], y_train_arr, weights)
        stump["feature"] = col
        if best_stump is None or stump["ig"] > best_stump["ig"]:
            best_stump = stump
    # Compute weighted error
    predictions = stump_prediction(X_train_scaled[best_stump["feature"]], best_stump)
    misclassified = predictions != y_train_arr
    error = weights[misclassified].sum()

    if error == 0 or error >= 0.5: # can't use this stump (perfect (alpha = inf) or worse than random (alpha < 0))
        print(f"Iteration {iteration + 1}: skipping stump (error={error:.4f}), stopping.")
        continue 
    # Compute alpha
    alpha = 0.5 * np.log((1 - error) / error)
    best_stump["alpha"] = alpha

    # Update sample weights 
    weights[misclassified]  *= np.exp(alpha)    # up-weight wrong samples
    weights[~misclassified] *= np.exp(-alpha)   # down-weight correct samples
    weights /= weights.sum()                    # re-normalise

    weak_learners.append(best_stump)
    print(f"Iteration {iteration + 1}: feature='{best_stump['feature']}' | "
          f"IG={best_stump['ig']:.4f} | error={error:.4f} | alpha={alpha:.4f}")
    val_acc, val_f1, _ = evaluate_model(X_val_scaled, y_val, weak_learners)
    print(f"  → Val Accuracy={val_acc:.4f}, Val F1={val_f1:.4f}")

    if val_acc > best_val_acc:
        best_val_acc       = val_acc
        no_improve_count   = 0
        best_weak_learners = list(weak_learners)   # save best snapshot
    else:
        no_improve_count += 1
        if no_improve_count >= early_stop_patience:
            print(f"\nEarly stopping at iteration {iteration + 1}: "
                  f"no val improvement for {early_stop_patience} rounds.")
            break

print("\n=== Final Evaluation on Test Set ===")
print(f"Using best ensemble snapshot ({len(best_weak_learners)} stumps, "
      f"chosen by val accuracy={best_val_acc:.4f})\n")

test_acc, test_f1, test_cm = evaluate_model(X_test_scaled, y_test, best_weak_learners)
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test F1-Score : {test_f1:.4f}")
print("Confusion Matrix:")
print(test_cm)
print("\nClassification Report:")
print(classification_report(y_test, ada_predict(X_test_scaled, best_weak_learners)))

# Most confusing classes
off_diag = test_cm - np.diag(np.diag(test_cm))
conf_i, conf_j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
print(f"Most confusing: Class {conf_i} misclassified as Class {conf_j} "
      f"({off_diag[conf_i, conf_j]} times)")

# Confusion matrix plot
plt.figure(figsize=(7, 5))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title(f'AdaBoost – Confusion Matrix (Test Set)\nAcc={test_acc:.4f} | F1={test_f1:.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("adaboost_confusion_matrix.png", dpi=150)
plt.show()