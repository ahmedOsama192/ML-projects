
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC , abstractmethod

class bivariate_strategy(ABC) :

    @abstractmethod
    def analyze(self , df : pd.DataFrame , col1 : str , col2 : str) -> None :

        '''
        function takes dataframe and 2 columns to apply Bivariate analysis on them

        parameters :
        col1 : first column name
        col2 : second column name

        returns None
        '''

        pass


class numerical_with_numerical(bivariate_strategy) :

    
    def analyze(self , df : pd.DataFrame , col1 : str , col2 : str) -> None :

        '''
        function takes dataframe and 2 columns to apply Bivariate analysis(scatter Plot) on them

        parameters :
        col1 : first column name
        col2 : second column name

        returns : None
        '''

        sns.scatterplot(x= df[col1] , y = df[col2])
        plt.title(f'ScatterPlot between {col1} and {col2}')
        plt.tight_layout()
        plt.show()


class categorical_with_numerical(bivariate_strategy) :

    
    def analyze(self , df : pd.DataFrame , col1 : str , col2 : str) -> None :

        '''
        function takes dataframe and 2 columns to apply Bivariate analysis(scatter Plot) on them

        parameters :
        col1 : first column name
        col2 : second column name

        returns : None
        '''
        plt.figure(figsize=(10,6))
        sns.boxplot(x= df[col1] , y = df[col2],)
        plt.title(f'boxplot between {col1} and {col2}')
        plt.tick_params(axis='x' , rotation = 45)
        plt.tight_layout()
        plt.show()

class bivariate_analyzer() :

    def __init__(self , strategy) : 

        self.strategy = strategy
        
    def set_strategy(self , strategy) :

        '''
        this function allows to modify the strategy .. instead of makeing other objects

        returns : None
        '''

        self.strategy = strategy


    def execute_analysis(self , df , col1 , col2) :

        '''
        this function calls the analyze function for the specified strategy

        returns : None
        '''

        self.strategy.analyze(df , col1 , col2)

if __name__ == '__main__' :

    categorical_cols = ['Neighborhood' , 'SaleCondition' , 'MasVnrType']
    bi_analyzer = bivariate_analyzer(numerical_with_numerical())
    df = pd.read_csv('d:/myfiles/MachineLearning/pandas/My_Note_Book_Projects/Price_Prediction_System/data/train.csv')
    bi_analyzer.set_strategy(categorical_with_numerical())
    for col in categorical_cols :
        bi_analyzer.execute_analysis(df , col , 'SalePrice')



