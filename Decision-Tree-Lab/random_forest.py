from bagging import BaggingClassifier

class RandomForestClassifier(BaggingClassifier):
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, 
                 min_impurity_decrease=0.0, max_features="sqrt", random_seed=42):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            random_seed=random_seed
        )
