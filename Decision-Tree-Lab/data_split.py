import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path, seed=42):
    # load dataset
    df = pd.read_csv(file_path)

    # features and label
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    # Count number of rows in each class
    class_counts = y.value_counts().sort_index()
    print("Number of rows in each class:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count}")

    # 1) split into train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=seed
    )

    # 2) split temp into validation (10%) and test (20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=2/3,
        stratify=y_temp,
        random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test