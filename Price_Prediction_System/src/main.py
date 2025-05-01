
import json
import joblib
import warnings

from get_data import load_data 
from cleaning_and_scaling import preprocess_data
from Encoding import encode_data
from training import train_model

warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == '__main__' :

    df = load_data('/myfiles/MachineLearning/pandas/My_Note_Book_Projects/Price_Prediction_System/Data/train.csv' ,
              train_size=.7)
    
    df = preprocess_data(df)
    
    df = encode_data(df)
    # df.drop(columns='type' , inplace= True)

    model , metrics = train_model(df)# (RandomForestRegressor(random_state=42), {'RMSE': 30584.357015744157, 'R2': 0.8813054147901871})


    # Save the model to a file
    joblib.dump(model, 'random_forest_model.pkl')

    # Save metrics to a JSON file
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f)


    

