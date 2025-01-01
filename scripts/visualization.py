# visualization

import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(data, column, output_path=None):
    """
    Plot and save the distribution of a specific column.

    Args:
        data (pd.DataFrame): Dataset.
        column (str): Column to visualize.
        output_path (str): Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, color="blue")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved at {output_path}")
    plt.show()

def plot_correlation_matrix(data, output_path=None):
    """
    Plot and save the correlation matrix of the dataset.

    Args:
        data (pd.DataFrame): Dataset.
        output_path (str): Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(12, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")

    if output_path:
        plt.savefig(output_path)
        print(f"Correlation matrix saved at {output_path}")
    plt.show()
