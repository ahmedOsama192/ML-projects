

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from abc import ABC , abstractmethod


class univeriate_interface(ABC) :

    # every class will inherit from this class should implement this function
    @abstractmethod 
    def analyze(self , df : pd.DataFrame , cols : list[str]) -> None :

        '''
        this function is used to show univariate analysis on a single column 

        parameters

        df : the dataframe 
        col : column to be analyzed(col) 

        reutrn : None
        '''
        pass

class numerical_univariate_analysis(univeriate_interface) :

    def analyze(self, df : pd.DataFrame , cols : list[str]) -> None :

        '''
        this function is used to show univariate analysis(hist plot) on a single numerical column 

        parameters

        df : the dataframe 
        cols : Numerical columns to be analyzed

        reutrn : None
        '''
        

        n_cols = 3  # 3 plots per row
        n_rows = int(np.ceil(len(cols) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols):

            sns.histplot(df[col], bins=30, kde=True , ax = axes[i] ,edgecolor = 'black')
            axes[i].set_title(f'Distribution of {col}\nSkweness = {df[col].skew()}')
            
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()



class categorical_univariate_analysis(univeriate_interface) :

    def analyze(self, df : pd.DataFrame  , cols : list[str]) -> None :

        '''
        this function is used to show univariate analysis(count plot) on a single categorical column 

        parameters

        df : the dataframe 
        cols : categorical columns to be analyzed

        reutrn : None
        '''
        
        # plt.figure(figsize=(8,6))
        # sns.countplot(df[cols])
        # plt.title(f'count plot of col{cols}\n')
        # plt.show()

        n_cols = 2  # 2 plots per row
        n_rows = int(np.ceil(len(cols) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols):

            sns.countplot(x = df[col], ax = axes[i] , palette="muted")
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_title(f'countplot of {col}')
            
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


class univariate_analyzer() :

    def __init__(self , strategy ):

        '''
        this function takes the univariate strategy that we will apply later

        strategy : univariate strategy that we will apply

        return : None 
        '''

        self.strategy = strategy

    def set_strategy(self , strategy) :

        '''
        this function allows to modify the strategy later .. its important to do this instead of creating
        multiple objects

        strategy : univariate strategy that we will apply

        return : None 

        '''
        self.strategy = strategy

    def execute_analysis(self , df : pd.DataFrame , cols : list[str]) -> None:

        '''
        this function calls the analyze function from the class we specify .. 

        returns : None
        '''
        self.strategy.analyze(df , cols)




