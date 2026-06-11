from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from scipy.io import arff


def load_data():
    data, meta = arff.loadarff("adult.arff")
    df = pd.DataFrame(data)
    # replace missing values
    df = df.replace("?", "Unknown")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
    return df


def split_data(target_col, df):
    X = df.drop(columns=[target_col])
    y = (df[target_col] == ">50K").astype(int)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train_raw, X_test_raw, y_train, y_test


def transform_data(continuous_cols, categorical_cols, X_train_raw, X_test_raw):
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), continuous_cols),
            (
                "cat",
                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore"
                ),
                categorical_cols
            ),
        ]
    )
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    return preprocessor, X_train, X_test


def process_data(target_col, continuous_cols, categorical_cols):
    df = load_data()
    X_train_raw, X_test_raw, y_train, y_test = split_data(target_col, df)
    preprocessor, X_train, X_test = transform_data(continuous_cols, categorical_cols, X_train_raw, X_test_raw)
    return preprocessor, X_train, X_test, y_train, y_test