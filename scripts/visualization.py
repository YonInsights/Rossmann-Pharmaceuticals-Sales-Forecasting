import seaborn as sns
import matplotlib.pyplot as plt
import logging

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
