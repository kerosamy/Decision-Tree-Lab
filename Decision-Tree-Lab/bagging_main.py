import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from data_split import split_data
from feature_preprocessing import scale_numeric_features
from evaluate_model import evaluate_model, plot_confusion_matrix
from decision_tree import DecisionTree
from bagging import BaggingClassifier
from random_forest import RandomForestClassifier

def run_tuning(model_class, grid, X_train, y_train, X_val, y_val):
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_score = -1
    best_params = None
    best_model = None
    
    print(f"Tuning {model_class.__name__}...")
    for params in combinations:
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        metrics, _ = evaluate_model(model, X_val, y_val, verbose=False)
        score = metrics["F1-Macro"]
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
            
    return best_model, best_params

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = split_data("../data/heart.csv")
    X_train, X_val, X_test, _ = scale_numeric_features(X_train, X_val, X_test)

    # Search spaces
    bagging_grid = {"n_estimators": [10, 20, 50], "max_depth": [5, 10], "min_samples_split": [5, 10]}
    rf_grid = {"n_estimators": [10, 20, 50], "max_depth": [5, 10], "min_samples_split": [5, 10], "max_features": ["sqrt"]}

    # Tune
    best_bag, bag_p = run_tuning(BaggingClassifier, bagging_grid, X_train, y_train, X_val, y_val)
    best_rf, rf_p = run_tuning(RandomForestClassifier, rf_grid, X_train, y_train, X_val, y_val)

    # Final results
    models = {"Bagging": best_bag, "RF": best_rf}
    print("\nTest Results:")
    for name, m in models.items():
        metrics, y_pred = evaluate_model(m, X_test, y_test, verbose=False)
        print(f"{name}: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1-Macro']:.4f}")
        
        plot_confusion_matrix(y_test, y_pred, title=f"{name} Results")
        plt.savefig(f"{name.lower()}_cm.png")
        plt.close()