import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize(df_clean):
    df_clean = df_clean.drop(columns=['no.'], errors='ignore')
    sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
    plt.rcParams["figure.figsize"] = (10, 6)

    # Distribution plots
    process_params = ['diameter_(mm)', 'speed_(rpm)', 'feed_(mm/rev)']
    for col in process_params:
        plt.figure()
        sns.histplot(df_clean[col], kde=True, bins=30)
        plt.title(f'Distribution of {col.replace("_", " ").title()}')
        plt.xlabel(col.replace("_", " ").title())
        plt.ylabel('Frequency')
        plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    corr = df_clean.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()

    # Outlier boxplots
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=df_clean[col])
        plt.title(f'Outlier Detection - {col.replace("_", " ").title()}')
        plt.show()

    # Trend Analysis
    sns.lmplot(data=df_clean, x='speed_(rpm)', y='flank_wear_(mm)', height=6, aspect=1.5)
    plt.title("Flank Wear vs Speed")
    plt.show()

    sns.lmplot(data=df_clean, x='feed_(mm/rev)', y='flank_wear_(mm)', height=6, aspect=1.5)
    plt.title("Flank Wear vs Feed")
    plt.show()

    # Barplot: Mean flank wear by binned speed
    df_clean['speed_bin'] = pd.cut(df_clean['speed_(rpm)'], bins=5)
    plt.figure()
    sns.barplot(data=df_clean, x='speed_bin', y='flank_wear_(mm)', estimator=np.mean)
    plt.xticks(rotation=45)
    plt.title("Mean Flank Wear by Speed Bin")
    plt.xlabel("Speed (binned)")
    plt.ylabel("Mean Flank Wear (mm)")
    plt.show()

    # Countplot: Binned diameter
    df_clean['diameter_bin'] = pd.cut(df_clean['diameter_(mm)'], bins=5)
    plt.figure()
    sns.countplot(data=df_clean, x='diameter_bin')
    plt.xticks(rotation=45)
    plt.title("Count of Observations by Diameter Bin")
    plt.xlabel("Diameter (binned)")
    plt.ylabel("Count")
    plt.show()

    # Violin Plot: Flank wear vs speed bin
    plt.figure()
    sns.violinplot(data=df_clean, x='speed_bin', y='flank_wear_(mm)', inner='quart')
    plt.xticks(rotation=45)
    plt.title("Flank Wear Distribution Across Speed Bins")
    plt.show()

    # Violin Plot: Flank wear vs diameter bin
    plt.figure()
    sns.violinplot(data=df_clean, x='diameter_bin', y='flank_wear_(mm)', inner='quart')
    plt.xticks(rotation=45)
    plt.title("Flank Wear Distribution Across Diameter Bins")
    plt.show()

    # Clean up temp bin columns
    df_clean.drop(columns=['speed_bin', 'diameter_bin'], inplace=True, errors='ignore')
