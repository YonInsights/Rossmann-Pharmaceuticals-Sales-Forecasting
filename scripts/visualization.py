import seaborn as sns
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
def plot_sales_distribution(train_data, hist_bins=30, hist_title='Sales Distribution - Histogram', 
                            density_title='Sales Distribution - Density Plot', x_label='Sales'):
    """
    Plots the sales distribution using both histogram and density plot.

    Parameters:
    train_data (DataFrame): The dataset containing sales data.
    hist_bins (int): Number of bins for the histogram. Default is 30.
    hist_title (str): Title for the histogram plot. Default is 'Sales Distribution - Histogram'.
    density_title (str): Title for the density plot. Default is 'Sales Distribution - Density Plot'.
    x_label (str): Label for the x-axis. Default is 'Sales'.

    Returns:
    None
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Setting visualization style.")
    # Set the style of the visualization
    sns.set(style="whitegrid")

    logger.info("Creating figure with subplots.")
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    logger.info("Plotting histogram of sales.")
    # Plot histogram of sales
    sns.histplot(train_data['Sales'], bins=hist_bins, kde=False, ax=ax[0])
    ax[0].set_title(hist_title)
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel('Frequency')

    logger.info("Plotting density plot of sales.")
    # Plot density plot of sales
    sns.kdeplot(train_data['Sales'], fill=True, ax=ax[1])
    ax[1].set_title(density_title)
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel('Density')

    logger.info("Displaying the plots.")
    # Show the plots
    plt.tight_layout()
    plt.show()

def plot_sales_over_time(data, date_column='Date', sales_column='Sales', 
                         title='Sales Over Time', x_label='Date', y_label='Sales'):
    """
    Plots the sales over time.

    Parameters:
    data (DataFrame): The dataset containing sales data.
    date_column (str): The name of the date column in the dataset. Default is 'Date'.
    sales_column (str): The name of the sales column in the dataset. Default is 'Sales'.
    title (str): The title of the plot. Default is 'Sales Over Time'.
    x_label (str): The label for the x-axis. Default is 'Date'.
    y_label (str): The label for the y-axis. Default is 'Sales'.

    Returns:
    None
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Converting the date column to datetime format.")
    # Convert the 'Date' column to datetime format
    data[date_column] = pd.to_datetime(data[date_column])

    logger.info("Grouping by date and calculating the mean sales.")
    # Group by date and calculate the mean sales
    sales_over_time = data.groupby(date_column)[sales_column].mean()

    logger.info("Creating the plot.")
    # Plot the sales over time
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(sales_over_time.index, sales_over_time.values, label='Sales')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    logger.info("Displaying the plot.")
    # Log the completion of the plot
    logging.info("Time series plot for sales over time has been created.")
    plt.show()
def plot_sales_by_store_type(train_data, store_data, store_column='Store', sales_column='Sales',
                             store_type_columns=['StoreType_b', 'StoreType_c', 'StoreType_d']):
    """
    Plots the sales distribution by store type.

    Parameters:
    train_data (DataFrame): The training dataset containing sales data.
    store_data (DataFrame): The dataset containing store types.
    store_column (str): The column name for store identifiers. Default is 'Store'.
    sales_column (str): The column name for sales data. Default is 'Sales'.
    store_type_columns (list): List of columns representing different store types.

    Returns:
    None
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Ensuring the store column in both dataframes has the same data type.")
    # Ensure the Store column in both dataframes has the same data type
    train_data[store_column] = train_data[store_column].astype(int)
    store_data[store_column] = store_data[store_column].astype(int)

    logger.info("Merging train_data with store_data to get store types.")
    # Merge train_data with store_data to get store types
    merged_data = pd.merge(train_data, store_data, on=store_column)

    # Check if merged_data is empty
    if merged_data.empty:
        logger.error("merged_data is empty. Please check the merge operation.")
        print("merged_data is empty. Please check the merge operation.")
    else:
        logger.info("Plotting the sales distribution by store type.")
        # Plot the sales distribution by store type
        plt.figure(figsize=(12, 6))
        for store_type_column in store_type_columns:
            sns.boxplot(x=store_type_column, y=sales_column, data=merged_data)
        plt.title('Sales Distribution by Store Type')
        plt.xlabel('Store Type')
        plt.ylabel(sales_column)
        plt.show()
        logger.info("Time series plot for sales by store type has been created.")

