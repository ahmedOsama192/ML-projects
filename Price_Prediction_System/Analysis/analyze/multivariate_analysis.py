

import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd


class multi_analysis :

    def analyze(self , df ) :

        '''
        this function just calls these 2 functions show_corr_map , show_pair_plot
        '''
        
        self.show_corr_map(df)
        self.show_pair_plot(df)


    def show_corr_map(self , df ) :

        """
        Generates and displays a correlation heatmap for the numerical features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a heatmap showing correlations between numerical features.
        """

        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def show_pair_plot(self , df) :

        """
        Generates and displays a pair plot for the selected features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a pair plot for the selected features.
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()

if __name__ =='__main__' :

    analyzer = multi_analysis()
    df = pd.read_csv('d:/myfiles/MachineLearning/pandas/My_Note_Book_Projects/Price_Prediction_System/Data/train.csv')

    selected_features = df[['SalePrice', 'GrLivArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt']]

    analyzer.analyze(selected_features)




