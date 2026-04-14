import numpy as np

class DecisionTree:
    class Node:
        def __init__(self, feature=None, threshold=None, is_categorical=False,
                     left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.is_categorical = is_categorical
            self.left = left
            self.right = right
            self.value = value

    def __init__(self, max_depth=10, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None

    # FIT
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.root = self._build_tree(self.X, self.y)

    # TREE BUILDING
    def _build_tree(self, X, y, depth=0):

        # stopping conditions
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return self.Node(value=self._most_common_label(y))

        best_feature, best_value, is_cat, best_gain = self._best_split(X, y)

        # no valid split or weak split
        if best_feature is None or best_gain < self.min_impurity_decrease:
            return self.Node(value=self._most_common_label(y))

        # split data
        if is_cat:
            left_idx = X[:, best_feature] == best_value
        else:
            left_idx = X[:, best_feature] <= best_value

        right_idx = ~left_idx

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return self.Node(best_feature, best_value, is_cat, left, right)

    # BEST SPLIT (INFORMATION GAIN)
    def _best_split(self, X, y):

        best_gain = -1
        best_feature = None
        best_value = None
        best_is_cat = False

        parent_entropy = self._entropy(y)
        n_samples, n_features = X.shape

        for feature in range(n_features):
            values = X[:, feature]
            unique_vals = np.unique(values)

            is_numeric = np.issubdtype(values.dtype, np.number)

            # NUMERIC FEATURE
            if is_numeric:
                for threshold in unique_vals:

                    gain = self._information_gain_numeric(
                        values, y, threshold, parent_entropy
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_value = threshold
                        best_is_cat = False

            # CATEGORICAL FEATURE
            else:
                for val in unique_vals:

                    gain = self._information_gain_cat(
                        values, y, val, parent_entropy
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_value = val
                        best_is_cat = True

        return best_feature, best_value, best_is_cat, best_gain

    # INFORMATION GAIN - NUMERIC
    def _information_gain_numeric(self, feature_values, y, threshold, parent_entropy):

        left = y[feature_values <= threshold]
        right = y[feature_values > threshold]

        if len(left) == 0 or len(right) == 0:
            return 0

        n = len(y)

        child_entropy = (
            (len(left) / n) * self._entropy(left) +
            (len(right) / n) * self._entropy(right)
        )

        return parent_entropy - child_entropy

    # INFORMATION GAIN - CATEGORICAL
    def _information_gain_cat(self, feature_values, y, value, parent_entropy):

        left = y[feature_values == value]
        right = y[feature_values != value]

        if len(left) == 0 or len(right) == 0:
            return 0

        n = len(y)

        child_entropy = (
            (len(left) / n) * self._entropy(left) +
            (len(right) / n) * self._entropy(right)
        )

        return parent_entropy - child_entropy

    # ENTROPY
    def _entropy(self, y):
        classes = np.unique(y)
        entropy = 0

        for c in classes:
            p = np.sum(y == c) / len(y)
            entropy -= p * np.log2(p + 1e-9)

        return entropy

    # MOST COMMON LABEL
    def _most_common_label(self, y):
        return max(set(y), key=list(y).count)

    # PREDICT
    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value

        if node.is_categorical:
            go_left = (x[node.feature] == node.threshold)
        else:
            go_left = (x[node.feature] <= node.threshold)

        if go_left:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)