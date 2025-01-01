import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


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
    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(12, 8))
    corr_matrix = numeric_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")

    if output_path:
        plt.savefig(output_path)
        print(f"Correlation matrix saved at {output_path}")
    plt.show()

def plot_timeseries(data, date_column, value_column, title="Time-Series Plot", output_path=None):
    """
    Plot and save a time-series graph for a specific column over time.

    Args:
        data (pd.DataFrame): Dataset.
        date_column (str): Column representing time.
        value_column (str): Column to visualize.
        title (str): Title of the plot.
        output_path (str): Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(10, 6))
    data[date_column] = pd.to_datetime(data[date_column])
    sns.lineplot(x=data[date_column], y=data[value_column], color="green")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(value_column)

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved at {output_path}")
    plt.show()

def plot_categorical_distribution(data, column, title="Categorical Distribution", output_path=None):
    """
    Plot and save the distribution of a categorical column.

    Args:
        data (pd.DataFrame): Dataset.
        column (str): Column to visualize.
        title (str): Title of the plot.
        output_path (str): Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data[column], palette="Set2")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved at {output_path}")
    plt.show()