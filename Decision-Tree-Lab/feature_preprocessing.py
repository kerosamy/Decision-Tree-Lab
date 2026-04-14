from sklearn.preprocessing import StandardScaler

def scale_numeric_features(X_train, X_val, X_test):
    # detect numeric columns
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    # initialize scaler
    scaler = StandardScaler()

    # fit only on training data
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    # transform validation and test
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_val, X_test, scaler