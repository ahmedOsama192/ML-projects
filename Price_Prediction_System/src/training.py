
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from get_data import load_data
from cleaning_and_scaling import preprocess_data
from Encoding import encode_data
import numpy as np

def train_model(df, target_col='SalePrice', model=None, random_state=42):
    """
    Trains a regression model on the given DataFrame.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame including features and target.
        target_col (str): Name of the target column.
        model (sklearn regressor): Regressor object (optional). Defaults to RandomForest.
        test_size (float): Fraction of data to use for validation.
        random_state (int): Seed for reproducibility.

    Returns:
        trained_model: The trained regression model.
        dict: Evaluation metrics on validation set.
    """
    
    # print(df)

        
    y_train = df[df['type'] == 'train'][target_col]
    y_test = df[df['type'] == 'test'][target_col]

    X_train = df[df['type'] == 'train'].drop(columns=['type', target_col])
    X_test = df[df['type'] == 'test'].drop(columns=['type', target_col])

    cols_to_drop = ['GarageYrBlt', 'Fence']

    X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
    X_test = X_test.drop(columns=cols_to_drop, errors='ignore')


    # print("X_train object columns:", X_train.select_dtypes(include='object').columns.tolist())
    # print("X_test object columns:", X_test.select_dtypes(include='object').columns.tolist())
    # print("y_train type:", y_train.dtype)
    # print("y_test type:", y_test.dtype)

    # for col in X_train.columns:
    #     bad_vals = X_train[col][~X_train[col].apply(lambda x: isinstance(x, (int, float, np.number)))]
    #     if not bad_vals.empty:
    #         print(f"Non-numeric values in column '{col}':")
    #         print(bad_vals.unique())


    # # Default model
    if model is None:
        model = RandomForestRegressor(random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred)
    }

    return model, metrics

if __name__ == '__main__' :

    df = load_data('/myfiles/MachineLearning/pandas/My_Note_Book_Projects/Price_Prediction_System/Data/train.csv' ,
              train_size=.7)
    
    df = preprocess_data(df)
    
    df = encode_data(df)

    print(train_model(df))
