import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_predictions(y_true, y_pred, target_names):
    for i, target in enumerate(target_names):
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_true[:, i], y=y_pred[:, i], alpha=0.6)
        plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                 [y_true[:, i].min(), y_true[:, i].max()], 'r--')
        plt.xlabel(f"Actual {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"{target} - Actual vs Predicted")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
