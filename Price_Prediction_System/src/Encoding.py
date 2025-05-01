import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from get_data import load_data 
from cleaning_and_scaling import preprocess_data


pd.set_option('future.no_silent_downcasting', True)  # Handle the FutureWarning

def encode_data_helper(df, target_col='SalePrice', is_train=True, 
                      global_target_means=None, onehot_columns=None):

    # ========= 1. Define encoding groups =========
    ordinal_features = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
        'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', 'LotShape', 'LandSlope',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish',
        'PavedDrive', 'Utilities', 'CentralAir'
    ]
    onehot_features = [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour',
        'LotConfig', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'Foundation', 'Heating', 'Electrical', 'GarageType', 'MiscFeature',
        'SaleType', 'SaleCondition'
    ]
    target_encode_features = ['Neighborhood', 'MoSold']

    # ========= 2. Ordinal Encoding =========
    ord_cols_present = [col for col in ordinal_features if col in df.columns]
    if ord_cols_present:
        # Handle missing values properly without downcasting warning
        df[ord_cols_present] = df[ord_cols_present].apply(
            lambda x: x.fillna('Missing') if x.dtype == 'object' else x
        )
        df[ord_cols_present] = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=np.nan
        ).fit_transform(df[ord_cols_present])

    # ========= 3. One-Hot Encoding =========
    onehot_present = [col for col in onehot_features if col in df.columns]
    if onehot_present:
        # Handle missing values properly
        df[onehot_present] = df[onehot_present].apply(
            lambda x: x.fillna('Missing') if x.dtype == 'object' else x
        )
        
        if is_train:
            # For training data - get all possible categories
            df = pd.get_dummies(df, columns=onehot_present, drop_first=True)
            onehot_columns = df.columns.tolist()
        else:
            # For test data - use same columns as training
            df = pd.get_dummies(df, columns=onehot_present, drop_first=False)
            
            # Add missing columns that exist in training but not in test
            missing_cols = set(onehot_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            
            # Remove extra columns that might exist in test but not training
            df = df[onehot_columns]

    # ========= 4. Target Encoding =========
    if is_train:
        global_target_means = {}
        for col in target_encode_features:
            if col in df.columns:
                # Calculate means with smoothing
                means = df.groupby(col)[target_col].mean()
                global_mean = df[target_col].mean()
                counts = df.groupby(col)[target_col].count()
                smooth_means = (counts * means + 10 * global_mean) / (counts + 10)
                
                df[f'{col}_TE'] = df[col].map(smooth_means)
                global_target_means[col] = smooth_means.to_dict()
        
        # Drop original columns
        df = df.drop(columns=target_encode_features, errors='ignore')
    else:
        for col in target_encode_features:
            if col in global_target_means:
                df[f'{col}_TE'] = df[col].map(global_target_means[col])
                df[f'{col}_TE'] = df[f'{col}_TE'].fillna(
                    np.mean(list(global_target_means[col].values()))
                )
                df = df.drop(columns=col, errors='ignore')

    # ========= 5. Convert bool to int =========
    bool_cols = df.select_dtypes(include='bool').columns
    if not bool_cols.empty:
        df[bool_cols] = df[bool_cols].astype(int)

    return df, global_target_means, onehot_columns if is_train else None


def encode_data(df) :    
    
    # Split data
    train_df = df[df['type'] == 'train'].drop(columns='type')
    test_df = df[df['type'] == 'test'].drop(columns='type')

    # Encode training data
    train_encoded, target_means, onehot_columns = encode_data_helper(
        train_df, is_train=True
    )

    # Encode test data using training parameters
    test_encoded, _, _ = encode_data_helper(
        test_df, 
        is_train=False, 
        global_target_means=target_means,
        onehot_columns=onehot_columns
    )

    # Combine data
    encoded_df = pd.concat([
        train_encoded.assign(type='train'),
        test_encoded.assign(type='test')
    ], ignore_index=True)

    # Final check for missing values
    # print("Missing values after encoding:")
    missing_values = encoded_df.isna().sum()
    # print(missing_values[missing_values > 0])
    
    if missing_values.sum() > 0:
        # Fill any remaining NAs (use median for numeric, mode for categorical)
        num_cols = encoded_df.select_dtypes(include=np.number).columns
        cat_cols = encoded_df.select_dtypes(exclude=np.number).columns
        
        encoded_df[num_cols] = encoded_df[num_cols].fillna(encoded_df[num_cols].median())

    return encoded_df

    # print(encoded_df)

if __name__ == '__main__':
    df = load_data('/myfiles/MachineLearning/pandas/My_Note_Book_Projects/Price_Prediction_System/Data/train.csv',
                   train_size=0.7)

    df = preprocess_data(df)

    df = encode_data(df)

    print(df.head())
      
# 
# # ===== Usage Example =====
# if __name__ == '__main__':
#     # Load and preprocess data
#     train = pd.read_csv('train.csv')
#     test = pd.read_csv('test.csv')
    
#     # Add identifier
#     train['type'] = 'train'
#     test['type'] = 'test'
#     combined = pd.concat([train, test])
    
#     # Clean data (using our previous function)
#     combined = handle_missing_values(combined)
    
#     # Split back
#     train_clean = combined[combined['type'] == 'train']
#     test_clean = combined[combined['type'] == 'test']
    
#     # Encode
#     train_encoded = encode_data(train_clean, target_col='SalePrice', is_train=True)
#     test_encoded = encode_data(test_clean, target_col='SalePrice', is_train=False)
    
#     print(f"Train shape: {train_encoded.shape}")
#     print(f"Test shape: {test_encoded.shape}")