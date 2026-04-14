from data_split import split_data
from feature_preprocessing import scale_numeric_features
from evaluate_model import evaluate_model, plot_confusion_matrix
from decision_tree import DecisionTree

# MAIN
if __name__ == "__main__":


    file_path = "data/heart.csv"

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(file_path)

    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)
    tree = DecisionTree(max_depth=100, min_samples_split=2, min_impurity_decrease=0.01)
    tree.fit(X_train, y_train)

    y_val_pred = tree.predict(X_val)
    y_test_pred = tree.predict(X_test)

    print("Validation Results:")
    evaluate_model(tree, X_val, y_val)
    plot_confusion_matrix(y_val, y_val_pred, title="Validation Confusion Matrix")

    print("Test Results:")
    evaluate_model(tree, X_test, y_test)
    plot_confusion_matrix(y_test, y_test_pred, title="Test Confusion Matrix")
    dot = tree.plot_tree(X_train.columns.tolist())
    dot.render("tree", format="png", view=True)
   