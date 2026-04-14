import numpy as np
from decision_tree import DecisionTree

class BaggingClassifier:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, 
                 min_impurity_decrease=0.0, max_features=None, random_seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_seed = random_seed
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        np.random.seed(self.random_seed)
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            y_bootstrap = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        y_pred = []
        for sample_idx in range(tree_preds.shape[1]):
            predictions = tree_preds[:, sample_idx]
            most_common = max(set(predictions), key=list(predictions).count)
            y_pred.append(most_common)
            
        return np.array(y_pred)
