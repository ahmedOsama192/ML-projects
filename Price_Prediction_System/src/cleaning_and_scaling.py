

import pandas as pd
from scipy.stats import skew
from get_data import load_data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


def handle_outliers(df, threshold=3.0):

    # Exclude target column
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop('SalePrice', errors='ignore')

    train_df = df[df['type'] == 'train']
    Q1 = train_df[num_cols].quantile(0.25)
    Q3 = train_df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    mask = ~((df[num_cols] < lower_bound) | (df[num_cols] > upper_bound)).any(axis=1)
    return df[mask].reset_index(drop=True)

def drop_duplicates(df, subset=None, keep='first'):
    return df.drop_duplicates(subset=subset, keep=keep)

def fix_skewness(df):
    from sklearn.preprocessing import PowerTransformer, StandardScaler

    # Exclude target from transformations
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop('SalePrice', errors='ignore')

    for col in numeric_cols:
        if abs(skew(df[col].dropna())) > 0.75:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            df[col] = pt.fit_transform(df[[col]]).flatten()

    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    return df

def preprocess_data(df):

    to_categorical = [
        'MSSubClass', 'OverallQual', 'OverallCond',
        'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold',
        'GarageYrBlt', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
        'Fireplaces', 'GarageCars'
    ]

    df[to_categorical] = df[to_categorical].astype('object')
    df.drop(columns=['Id'], inplace=True)

    df = handle_missing_values(df)
    df = fix_skewness(df)
    df = drop_duplicates(df)
    df = handle_outliers(df)

    return df

def handle_missing_values(df):

    train_mask = df['type'] == 'train'
    train_df = df[train_mask]

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("None")

    # Exclude SalePrice from imputation
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop('SalePrice', errors='ignore')

    if "LotFrontage" in num_cols and "Neighborhood" in cat_cols:
        lot_median = train_df.groupby("Neighborhood")["LotFrontage"].median()
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(lot_median[x.name] if x.name in lot_median else x.median()))

    for col in num_cols:
        if df[col].isna().any():
            median_val = train_df[col].median()
            df[col] = df[col].fillna(median_val)

    return df

if __name__ == '__main__':
    df = load_data('/myfiles/MachineLearning/pandas/My_Note_Book_Projects/Price_Prediction_System/Data/train.csv',
                   train_size=0.7)

    df = preprocess_data(df)





   
