

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Missing_values :

    def analyze(self , df : pd.DataFrame) -> None :

        print('Number of Nulls in each Column : \n')
        print(df.isna().sum())

        
    def Visualizing_Nans(self , df) :

        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isna(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title('Heatmap of Missing Values')
        # plt.xticks(rotation=90 , fontsize=12)
        plt.show()

        # ---- Bar Plot ----
        nans = df.isna().sum()
        nans = nans[nans > 0].sort_values(ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=nans.index.to_list() , y=nans.values, palette='magma')
        plt.title('Count of Missing Values per Column')
        # plt.ylabel('Missing Count')
        plt.xticks(rotation=90 , fontsize=12)
        plt.show()


        


if __name__ == '__main__' :
     
    df = pd.read_csv('Data/train.csv')
    Missing_val = Missing_values()
    Missing_val.Visualizing_Nans(df)
