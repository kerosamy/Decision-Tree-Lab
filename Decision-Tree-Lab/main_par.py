from data_split import split_data
from evaluate_model import evaluate_model, plot_confusion_matrix
from decision_tree import DecisionTree

import pandas as pd

# MAIN
if __name__ == "__main__":

    file_path = "data/heart.csv"

    # SPLIT DATA
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(file_path)

    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    # HYPERPARAMETER TUNING
    max_depth_list = [3, 5, 7, 10, 15]
    min_samples_split_list = [2, 5, 10, 20]
    min_impurity_list = [0.0, 1e-4, 1e-3, 1e-2]

    results = []

    best_score = -1
    best_model = None
    best_params = None

    print("\n🔍 Tuning models (F1-Macro)...\n")

    for depth in max_depth_list:
        for min_split in min_samples_split_list:
            for min_imp in min_impurity_list:

                tree = DecisionTree(
                    max_depth=depth,
                    min_samples_split=min_split,
                    min_impurity_decrease=min_imp
                )

                tree.fit(X_train, y_train)

                metrics, _ = evaluate_model(tree, X_val, y_val, verbose=False)

                f1_macro = metrics["F1-Macro"]

                results.append({
                    "max_depth": depth,
                    "min_samples_split": min_split,
                    "min_impurity": min_imp,
                    "f1_macro": f1_macro
                })

                print(f"depth={depth}, split={min_split}, imp={min_imp} → F1={f1_macro:.4f}")

                if f1_macro > best_score:
                    best_score = f1_macro
                    best_model = tree
                    best_params = (depth, min_split, min_imp)

    # RESULTS TABLE
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="f1_macro", ascending=False)

    print("\n=== 🔥 Tuning Results (Sorted by F1-Macro) ===")
    print(df_results.to_string(index=False))

    # BEST PARAMETERS
    print("\n=== BEST PARAMETERS ===")
    print(f"max_depth = {best_params[0]}")
    print(f"min_samples_split = {best_params[1]}")
    print(f"min_impurity_decrease = {best_params[2]}")
    print(f"Best Validation F1-Macro = {best_score:.4f}")

    # ==============================
    # TEST EVALUATION
    # ==============================
    print("\n===TEST RESULTS (BEST MODEL) ===")
    metrics, y_test_pred = evaluate_model(best_model, X_test, y_test)

    plot_confusion_matrix(y_test, y_test_pred, title="Best Model Confusion Matrix")

    # ==============================
    # TREE VISUALIZATION
    # ==============================
    print("\n Plotting Best Tree (limited depth)...")

    viz_tree = DecisionTree(
        max_depth=best_params[0],
        min_samples_split=best_params[1],
        min_impurity_decrease=best_params[2]
    )

    viz_tree.fit(X_train, y_train)

    dot = viz_tree.plot_tree(X_train.columns.tolist())
    dot.render("best_tree", format="png", view=True)