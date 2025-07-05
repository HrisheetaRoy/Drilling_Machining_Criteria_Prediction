import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize(df_clean):
     # Add this line at the beginning of the function
     df_clean = df_clean.drop(columns=['no.'], errors='ignore')
     sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
     plt.rcParams["figure.figsize"] = (10, 6)

     #Distribution of 3 Process Parameters
     process_params = ['diameter_(mm)', 'speed_(rpm)', 'feed_(mm/rev)']

     for col in process_params:
          plt.figure()
          sns.histplot(df_clean[col], kde=True, bins=30)
          plt.title(f'Distribution of {col.replace("_", " ").title()}')
          plt.xlabel(col.replace("_", " ").title())
          plt.ylabel('Frequency')
          plt.show()

     #Correlation Heatmap (for all numeric columns)
     plt.figure(figsize=(10, 6))
     corr = df_clean.corr(numeric_only=True)
     sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
     plt.title("Correlation Matrix of Numerical Features")
     plt.show()

     #Outlier Detection (Boxplots)
     numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns

     for col in numeric_cols:
          plt.figure()
          sns.boxplot(x=df_clean[col])
          plt.title(f'Outlier Detection - {col.replace("_", " ").title()}')
          plt.show()

     #Trend Analysis
     sns.lmplot(data=df_clean, x='speed_(rpm)', y='flank_wear_(mm)', height=6, aspect=1.5)
     plt.title("Flank Wear vs Speed")
     plt.show()

     sns.lmplot(data=df_clean, x='feed_(mm/rev)', y='flank_wear_(mm)', height=6, aspect=1.5)
     plt.title("Flank Wear vs Feed")
     plt.show()

