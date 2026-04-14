from graphviz import Digraph
import numpy as np

class DecisionTree:
    class Node:
        def __init__(self, feature=None, threshold=None, is_categorical=False,
                     left=None, right=None, value=None, entropy=None, gain=None, samples=None):
            self.feature = feature
            self.threshold = threshold
            self.is_categorical = is_categorical
            self.left = left
            self.right = right
            self.value = value
            self.entropy = entropy
            self.gain = gain
            self.samples = samples

    def __init__(self, max_depth=10, min_samples_split=2, min_impurity_decrease=0.0, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.root = self._build_tree(self.X, self.y)

    def _build_tree(self, X, y, depth=0):

        if len(y) == 0:
            return self.Node(value=None, entropy=0, samples={})

        # stopping conditions
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return self.Node(
                value=self._most_common_label(y),
                entropy=self._entropy(y),
                samples={int(c): int(np.sum(y == c)) for c in np.unique(y)}
            )

        best_feature, best_value, is_cat, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain < self.min_impurity_decrease:
            return self.Node(
                value=self._most_common_label(y),
                entropy=self._entropy(y),
                samples={int(c): int(np.sum(y == c)) for c in np.unique(y)}
            )

        if is_cat:
            left_idx = X[:, best_feature] == best_value
        else:
            left_idx = X[:, best_feature].astype(float) <= float(best_value)

        right_idx = ~left_idx

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        entropy = self._entropy(y)
        samples = {}
        for c in np.unique(y):
            count = np.sum(y == c)
            samples[int(c)] = int(count)

        return self.Node(best_feature, best_value, is_cat,
                         left, right,
                         value=None,
                         entropy=entropy,
                         gain=best_gain,
                         samples=samples)

    def _best_split(self, X, y):

        best_gain = -1
        best_feature = None
        best_value = None
        best_is_cat = False

        parent_entropy = self._entropy(y)
        n_samples, n_features = X.shape

        # Feature sampling for Random Forest
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        elif self.max_features == "sqrt":
            n_sub = int(np.sqrt(n_features))
            feature_indices = np.random.choice(n_features, n_sub, replace=False)
        elif self.max_features == "log2":
            n_sub = int(np.log2(n_features + 1))
            feature_indices = np.random.choice(n_features, n_sub, replace=False)
        elif isinstance(self.max_features, int):
            feature_indices = np.random.choice(n_features, min(self.max_features, n_features), replace=False)
        else:
            feature_indices = np.arange(n_features)

        for feature in feature_indices:
            values = X[:, feature]
            unique_vals = np.unique(values)
            is_numeric = False
            try:
                unique_vals = unique_vals.astype(float)
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False

            # NUMERIC FEATURE
            if is_numeric:
                values_numeric = values.astype(float)
                for threshold in unique_vals:
                    gain = self._information_gain(
                        values_numeric, y, threshold, parent_entropy, is_numeric=True
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_value = threshold
                        best_is_cat = False

            # CATEGORICAL FEATURE
            else:
                for val in unique_vals:
                    gain = self._information_gain(
                        values, y, val, parent_entropy, is_numeric=False
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_value = val
                        best_is_cat = True

        return best_feature, best_value, best_is_cat, best_gain

    def _information_gain(self, feature_values, y, threshold, parent_entropy, is_numeric=True):

        if is_numeric:
            left = y[feature_values <= threshold]
            right = y[feature_values > threshold]
        else:
            left = y[feature_values == threshold]
            right = y[feature_values != threshold]

        # invalid split
        if len(left) == 0 or len(right) == 0:
            return 0

        n = len(y)

        child_entropy = (
            (len(left) / n) * self._entropy(left) +
            (len(right) / n) * self._entropy(right)
        )

        return parent_entropy - child_entropy

    def _entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for c in classes:
            p = np.sum(y == c) / len(y)
            entropy -= p * np.log2(p + 1e-9)
        return entropy

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            pred = self._traverse(x, self.root)
            predictions.append(pred)

        return np.array(predictions)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value

        if node.is_categorical:
            go_left = (x[node.feature] == node.threshold)
        else:
            go_left = (float(x[node.feature]) <= float(node.threshold))

        if go_left:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def plot_tree(self, feature_names=None):
        dot = Digraph()

        def add_nodes(node, parent_id=None, edge_label=""):
            if node is None:
                return

            node_id = str(id(node))

            # Leaf
            if node.value is not None:
                label = f"Leaf\nclass = {node.value}\nentropy = {node.entropy:.4f}\nsamples = {node.samples}"
                color = "#a6cee3" if node.value == 0 else "#fb9a99"
            else:
                feature = node.feature
                if feature_names:
                    feature = feature_names[feature]
                if node.is_categorical:
                    condition = f"{feature} == {node.threshold}"
                else:
                    condition = f"{feature} <= {node.threshold:.3f}"

                label = f"{condition}\nIG = {node.gain:.4f}\nentropy = {node.entropy:.4f}\nsamples = {node.samples}"
                color = "#f0f0f0"

            dot.node(node_id, label, style="filled", fillcolor=color)

            if parent_id is not None:
                dot.edge(parent_id, node_id, label=edge_label)

            if node.value is None:
                add_nodes(node.left, node_id, "True")
                add_nodes(node.right, node_id, "False")

        add_nodes(self.root)
        return dot