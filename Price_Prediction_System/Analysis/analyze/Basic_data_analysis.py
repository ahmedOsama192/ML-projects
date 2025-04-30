

import pandas as pd


class Basic_analysis :

    def Analyze_(self , df : pd.DataFrame) -> None :             
    
        """
        this function print some basic analysis about the data 
        data type - Number of not nulls in the columns and summary statistics in the data

        parameters :
        df : dataframe we analyze

        Returns : None
        """        
    
              
        print('some Info about the data : \n')
        df.info()

        print('\n______________________________\n')
        print('summary statistics for Numerical Features : \n')
        print(df.describe())  
        
        print('\n______________________________\n')
        print('summary statistics for categorical Features : \n')
        print(df.describe(include='object'))  




