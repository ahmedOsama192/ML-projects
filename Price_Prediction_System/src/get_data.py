


import pandas as pd

def split_data(df , train_sz = .7 , random_state=42) :


    """
    Splits data into train, test sets.
    
    Parameters:
    - df: Input DataFrame
    - train_size: Proportion for training set (default 0.7)
    - random_state: Random seed for reproducibility
    
    Returns:
    - df after combining df_train , df_test
    """

    from sklearn.model_selection import train_test_split

    df_train , df_test = train_test_split(df , 
                                        train_size=train_sz , 
                                        shuffle= True ,
                                        random_state=42)
    
    df_train['type'] = 'train'
    df_test['type'] = 'test'

    df = pd.concat((df_train , df_test) ,axis=0)
    

    return df 


def load_data(file_path , train_size = .7) :

    """
    Loads the data.
    
    Parameters:
    - data_path: Path to CSV file (default 'Data/train.csv')
    - train_size: Proportion for training set
    
    Returns:
    - df after the split
    """

    df= pd.read_csv(file_path)

    df = split_data(df , train_size)

    return df

df = load_data(
        '/myfiles/MachineLearning/pandas/My_Note_Book_Projects/Price_Prediction_System/Data/train.csv',
        train_size=.7
    )






