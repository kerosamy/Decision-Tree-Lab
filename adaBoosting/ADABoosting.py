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
estimators_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
early_stop_patience_list = [3, 5, 7, 10, 15]  # Different early stopping patience values

tuning_results = {
    'n_estimators': [],
    'early_stop_patience': [],
    'train_acc': [],
    'val_acc': [],
    'test_acc': [],
    'train_f1': [],
    'val_f1': [],
    'test_f1': [],
    'actual_estimators_used': []  # How many estimators were actually used (may be less due to early stopping)
}
best_overall_model = None
best_overall_test_acc = 0.0
best_n_estimators = None
best_early_stop_patience = None

y_train_arr = np.asarray(y_train, dtype=int)

total_combinations = len(estimators_list) * len(early_stop_patience_list)
combination_count = 0

for n_est in estimators_list:
    for patience in early_stop_patience_list:
        combination_count += 1
        print(f"\n{'='*80}")
        print(f"Training combination {combination_count}/{total_combinations}: n_estimators={n_est}, early_stop_patience={patience}")
        print(f"{'='*80}")
        
        n_estimators        = n_est
        early_stop_patience = patience
        weights = np.ones(len(X_train_scaled), dtype=float) / len(X_train_scaled)
        best_val_acc       = 0.0
        no_improve_count   = 0
        weak_learners      = []
        best_weak_learners = []
        actual_estimators_used = 0

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

            if error == 0 or error >= 0.5:
                print(f"Iteration {iteration + 1}: skipping stump (error={error:.4f}), stopping.")
                break 
            # Compute alpha
            alpha = 0.5 * np.log((1 - error) / error)
            best_stump["alpha"] = alpha

            # Update sample weights 
            weights[misclassified]  *= np.exp(alpha)
            weights[~misclassified] *= np.exp(-alpha)
            weights /= weights.sum()

            weak_learners.append(best_stump)
            actual_estimators_used += 1
            val_acc, val_f1, _ = evaluate_model(X_val_scaled, y_val, weak_learners)

            if val_acc > best_val_acc:
                best_val_acc       = val_acc
                no_improve_count   = 0
                best_weak_learners = list(weak_learners)
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    print(f"Early stopping at iteration {iteration + 1}: no val improvement for {early_stop_patience} rounds.")
                    break
        
        # Evaluate the model with the best validation performance on all sets
        train_acc, train_f1, _ = evaluate_model(X_train_scaled, y_train, best_weak_learners)
        val_acc, val_f1, _     = evaluate_model(X_val_scaled, y_val, best_weak_learners)
        test_acc, test_f1, _   = evaluate_model(X_test_scaled, y_test, best_weak_learners)
        
        tuning_results['n_estimators'].append(n_est)
        tuning_results['early_stop_patience'].append(patience)
        tuning_results['train_acc'].append(train_acc)
        tuning_results['val_acc'].append(val_acc)
        tuning_results['test_acc'].append(test_acc)
        tuning_results['train_f1'].append(train_f1)
        tuning_results['val_f1'].append(val_f1)
        tuning_results['test_f1'].append(test_f1)
        tuning_results['actual_estimators_used'].append(len(best_weak_learners))
        
        print(f"Result: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Test Acc={test_acc:.4f}")
        print(f"         Actual estimators used: {len(best_weak_learners)} (early stopped from {n_est})") 
        # Track the best model
        if test_acc > best_overall_test_acc:
            best_overall_test_acc = test_acc
            best_overall_model = best_weak_learners
            best_n_estimators = n_est
            best_early_stop_patience = patience

print(f"\n{'='*80}")
print(f"BEST MODEL: n_estimators={best_n_estimators}, early_stop_patience={best_early_stop_patience}")
print(f"Test Accuracy = {best_overall_test_acc:.4f}")
print(f"{'='*80}\n")

print("=== Final Evaluation on Test Set ===")
print(f"Using best model with n_estimators={best_n_estimators}, early_stop_patience={best_early_stop_patience}")
print(f"chosen by test accuracy={best_overall_test_acc:.4f}\n")

test_acc, test_f1, test_cm = evaluate_model(X_test_scaled, y_test, best_overall_model)
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test F1-Score : {test_f1:.4f}")
print("Confusion Matrix:")
print(test_cm)
print("\nClassification Report:")
print(classification_report(y_test, ada_predict(X_test_scaled, best_overall_model)))

# Most confusing classes
off_diag = test_cm - np.diag(np.diag(test_cm))
conf_i, conf_j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
print(f"Most confusing: Class {conf_i} misclassified as Class {conf_j} "
      f"({off_diag[conf_i, conf_j]} times)")

# Create pivot tables for heatmap visualization
results_df = pd.DataFrame(tuning_results)
acc_pivot = results_df.pivot(index='n_estimators', columns='early_stop_patience', values='test_acc')
f1_pivot = results_df.pivot(index='n_estimators', columns='early_stop_patience', values='test_f1')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Test Accuracy Heatmap
sns.heatmap(acc_pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Test Accuracy'})
axes[0].set_title('AdaBoost: Test Accuracy\n(n_estimators vs early_stop_patience)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Early Stop Patience', fontsize=12)
axes[0].set_ylabel('Max Estimators', fontsize=12)

# Plot 2: Test F1-Score Heatmap
sns.heatmap(f1_pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Test F1-Score'})
axes[1].set_title('AdaBoost: Test F1-Score\n(n_estimators vs early_stop_patience)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Early Stop Patience', fontsize=12)
axes[1].set_ylabel('Max Estimators', fontsize=12)

# Plot 3: Actual Estimators Used Heatmap
actual_est_pivot = results_df.pivot(index='n_estimators', columns='early_stop_patience', values='actual_estimators_used')
sns.heatmap(actual_est_pivot, annot=True, fmt='.0f', cmap='Blues', ax=axes[2], cbar_kws={'label': 'Actual Estimators Used'})
axes[2].set_title('AdaBoost: Actual Estimators Used\n(n_estimators vs early_stop_patience)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Early Stop Patience', fontsize=12)
axes[2].set_ylabel('Max Estimators', fontsize=12)

plt.tight_layout()
plt.savefig("adaboost_tuning_heatmap.png", dpi=150, bbox_inches='tight')
print("\nTuning heatmap saved to: adaboost_tuning_heatmap.png")
plt.show()

# Additional visualization: Line plots for each early_stop_patience value
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Group by early_stop_patience and plot lines
patience_values = sorted(list(set(tuning_results['early_stop_patience'])))
colors = plt.cm.tab10(np.linspace(0, 1, len(patience_values)))

for i, patience in enumerate(patience_values):
    mask = np.array(tuning_results['early_stop_patience']) == patience
    n_est_subset = np.array(tuning_results['n_estimators'])[mask]
    test_acc_subset = np.array(tuning_results['test_acc'])[mask]
    
    axes[0].plot(n_est_subset, test_acc_subset, 
                marker='o', label=f'Patience={patience}', 
                color=colors[i], linewidth=2, markersize=6)

axes[0].axvline(x=best_n_estimators, color='red', linestyle='--', linewidth=2, 
               label=f'Best: n_est={best_n_estimators}, pat={best_early_stop_patience}')
axes[0].set_xlabel('Number of Estimators', fontsize=12)
axes[0].set_ylabel('Test Accuracy', fontsize=12)
axes[0].set_title('AdaBoost: Test Accuracy vs Estimators\n(by Early Stop Patience)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(estimators_list)

# F1-Score plot
for i, patience in enumerate(patience_values):
    mask = np.array(tuning_results['early_stop_patience']) == patience
    n_est_subset = np.array(tuning_results['n_estimators'])[mask]
    test_f1_subset = np.array(tuning_results['test_f1'])[mask]
    
    axes[1].plot(n_est_subset, test_f1_subset, 
                marker='s', label=f'Patience={patience}', 
                color=colors[i], linewidth=2, markersize=6)

axes[1].axvline(x=best_n_estimators, color='red', linestyle='--', linewidth=2, 
               label=f'Best: n_est={best_n_estimators}, pat={best_early_stop_patience}')
axes[1].set_xlabel('Number of Estimators', fontsize=12)
axes[1].set_ylabel('Test F1-Score', fontsize=12)
axes[1].set_title('AdaBoost: Test F1-Score vs Estimators\n(by Early Stop Patience)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(estimators_list)

plt.tight_layout()
plt.savefig("adaboost_tuning_lines.png", dpi=150, bbox_inches='tight')
print("Tuning line plots saved to: adaboost_tuning_lines.png")
plt.show()

# Confusion matrix plot for the best model
plt.figure(figsize=(7, 5))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title(f'AdaBoost – Confusion Matrix (Test Set)\nBest: n_est={best_n_estimators}, pat={best_early_stop_patience}, Acc={test_acc:.4f}, F1={test_f1:.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("adaboost_confusion_matrix.png", dpi=150)
print("Confusion matrix saved to: adaboost_confusion_matrix.png")
plt.show()

# Print tuning summary table
print("\n" + "="*100)
print("TUNING SUMMARY (Grid Search Results)")
print("="*100)
summary_df = pd.DataFrame(tuning_results)
summary_df = summary_df.sort_values(['test_acc', 'test_f1'], ascending=[False, False])
print(summary_df.to_string(index=False))
print("="*100)
print(f"\nBEST CONFIGURATION:")
print(f"  n_estimators: {best_n_estimators}")
print(f"  early_stop_patience: {best_early_stop_patience}")
print(f"  Test Accuracy: {best_overall_test_acc:.4f}")
print(f"  Actual estimators used: {len(best_overall_model)}")
print("="*100)