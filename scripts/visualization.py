import seaborn as sns
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_average_sales(train_data, promo_col='Promo', sales_col='Sales', title='Average Sales on Promo Days vs. Non-Promo Days'):
    """
    Plots the average sales for promotional and non-promotional days.

    Parameters:
        train_data (pd.DataFrame): The dataframe containing the sales data.
        promo_col (str): The name of the column indicating promotional days. Default is 'Promo'.
        sales_col (str): The name of the sales column. Default is 'Sales'.
        title (str): The title of the plot. Default is 'Average Sales on Promo Days vs. Non-Promo Days'.
    """
    # Group by 'Promo' and calculate average sales
    promo_sales = train_data.groupby(promo_col)[sales_col].mean().reset_index()
    promo_sales.columns = ['Promo', 'Average Sales']

    # Log the calculated average sales
    logging.info(f"Calculated average sales: {promo_sales}")

    # Plotting the bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Promo', y='Average Sales', data=promo_sales, palette='viridis')
    plt.title(title, fontsize=14)
    plt.xlabel(f'{promo_col} (1 = Promo Day, 0 = Non-Promo Day)', fontsize=12)
    plt.ylabel('Average Sales', fontsize=12)
    plt.xticks([0, 1], ['Non-Promo Days', 'Promo Days'], fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
def plot_sales_trends(train_data):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Ensure `Date` is in datetime format
    logging.info("Converting 'Date' to datetime format...")
    train_data['Date'] = pd.to_datetime(train_data['Date'])

    # Create a column for holiday status
    logging.info("Creating 'Holiday_Status' column...")
    train_data['Holiday_Status'] = 'Non-Holiday'
    train_data.loc[train_data['StateHoliday'] != '0', 'Holiday_Status'] = 'During Holiday'

    # Sorting data by date
    logging.info("Sorting data by 'Date'...")
    train_data = train_data.sort_values(by='Date')

    # Aggregating sales by date and holiday status
    logging.info("Aggregating sales by 'Date' and 'Holiday_Status'...")
    holiday_sales = train_data.groupby(['Date', 'Holiday_Status'])['Sales'].mean().reset_index()

    # Plotting the sales trends
    logging.info("Plotting sales trends...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=holiday_sales, x='Date', y='Sales', hue='Holiday_Status', palette='Set2')
    plt.title('Sales Trends Before, During, and After Holidays', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Sales', fontsize=12)
    plt.legend(title='Holiday Status', loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    logging.info("Displaying the plot...")
    # Show the plot
    plt.show()

def calculate_promo_sales_summary(data):
    """
    Calculate mean sales grouped by Promo status.

    Args:
        data (DataFrame): The input dataset containing 'Promo' and 'Sales' columns.

    Returns:
        DataFrame: Summary of mean sales grouped by Promo.
    """
    return data.groupby("Promo")["Sales"].mean()

def plot_bar_chart(data):
    """
    Plot a bar chart to show average sales by Promo status.

    Args:
        data (DataFrame): The input dataset containing 'Promo' and 'Sales' columns.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Promo", y="Sales", data=data, palette="viridis")
    plt.title("Impact of Promotions on Sales")
    plt.xlabel("Promotion Status")
    plt.ylabel("Average Sales")
    plt.show()

def plot_boxplot(data):
    """
    Plot a boxplot to show the distribution of sales by Promo status.

    Args:
        data (DataFrame): The input dataset containing 'Promo' and 'Sales' columns.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Promo", y="Sales", data=data, palette="viridis")
    plt.title("Sales Distribution: Promotion vs. No Promotion")
    plt.xlabel("Promotion Status")
    plt.ylabel("Sales")
    plt.show()
def get_numerical_features_present(numerical_features, dataframe):
    """Filter the numerical features to include only those present in the DataFrame."""
    return [feature for feature in numerical_features if feature in dataframe.columns]

def calculate_correlation_matrix(dataframe, features):
    """Calculate the correlation matrix for the given features in the DataFrame."""
    return dataframe[features].corr()

def generate_heatmap(correlation_matrix, figsize=(10, 8), cmap='coolwarm', fmt='.2f', title='Correlation Matrix of Numerical Features'):
    """Generate a heatmap for the correlation matrix."""
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=fmt)
    plt.title(title)
    plt.show()
def prepare_data(data):
    """Prepare the data by converting the 'Date' column to datetime and setting it as the index."""
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def decompose_series(data, column, model='additive', period=365):
    """Decompose the time series data."""
    return seasonal_decompose(data[column], model=model, period=period)

def plot_decomposition(decomposition):
    """Plot the decomposed components of the time series."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    decomposition.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    plt.show()
def calculate_sales_growth(data):
    """Calculate the percentage growth in sales over time."""
    data['SalesGrowth'] = data['Sales'].pct_change() * 100
    return data

def plot_sales_growth(data):
    """Plot the percentage growth in sales over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['SalesGrowth'], label='Sales Growth')
    plt.xlabel('Date')
    plt.ylabel('Percentage Growth')
    plt.title('Percentage Growth in Sales Over Time')
    plt.legend()
    plt.show()

