import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

def plot_temperature_trend(data, date_column, temp_column, chart_type='line'):
    """
    Plot temperature trends over time.
    
    Parameters:
        data (pd.DataFrame): Weather data
        date_column (str): Name of the date column
        temp_column (str): Name of the temperature column
        chart_type (str): Type of chart (line, bar, area)
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort data by date
    data = data.sort_values(by=date_column)
    
    # Plot based on chart type
    if chart_type == 'line':
        ax.plot(data[date_column], data[temp_column], marker='', linestyle='-', linewidth=2, color='red')
    elif chart_type == 'bar':
        ax.bar(data[date_column], data[temp_column], color='red', alpha=0.7)
    elif chart_type == 'area':
        ax.fill_between(data[date_column], data[temp_column], color='red', alpha=0.5)
    
    # Add titles and labels
    ax.set_title('Temperature Trend Over Time', fontsize=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature', fontsize=12)
    
    # Improve visual appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.autofmt_xdate()  # Rotate date labels
    
    plt.tight_layout()
    return fig

def plot_humidity_trend(data, date_column, humid_column, chart_type='line'):
    """
    Plot humidity trends over time.
    
    Parameters:
        data (pd.DataFrame): Weather data
        date_column (str): Name of the date column
        humid_column (str): Name of the humidity column
        chart_type (str): Type of chart (line, bar, area)
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort data by date
    data = data.sort_values(by=date_column)
    
    # Plot based on chart type
    if chart_type == 'line':
        ax.plot(data[date_column], data[humid_column], marker='', linestyle='-', linewidth=2, color='blue')
    elif chart_type == 'bar':
        ax.bar(data[date_column], data[humid_column], color='blue', alpha=0.7)
    elif chart_type == 'area':
        ax.fill_between(data[date_column], data[humid_column], color='blue', alpha=0.5)
    
    # Add titles and labels
    ax.set_title('Humidity Trend Over Time', fontsize=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Humidity (%)', fontsize=12)
    
    # Improve visual appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.autofmt_xdate()  # Rotate date labels
    
    plt.tight_layout()
    return fig

def plot_pressure_trend(data, date_column, pressure_column, chart_type='line'):
    """
    Plot pressure trends over time.
    
    Parameters:
        data (pd.DataFrame): Weather data
        date_column (str): Name of the date column
        pressure_column (str): Name of the pressure column
        chart_type (str): Type of chart (line, bar, area)
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort data by date
    data = data.sort_values(by=date_column)
    
    # Plot based on chart type
    if chart_type == 'line':
        ax.plot(data[date_column], data[pressure_column], marker='', linestyle='-', linewidth=2, color='purple')
    elif chart_type == 'bar':
        ax.bar(data[date_column], data[pressure_column], color='purple', alpha=0.7)
    elif chart_type == 'area':
        ax.fill_between(data[date_column], data[pressure_column], color='purple', alpha=0.5)
    
    # Add titles and labels
    ax.set_title('Atmospheric Pressure Trend Over Time', fontsize=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Pressure', fontsize=12)
    
    # Improve visual appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.autofmt_xdate()  # Rotate date labels
    
    plt.tight_layout()
    return fig

def plot_rainfall_trend(data, date_column, rainfall_column, chart_type='bar'):
    """
    Plot rainfall trends over time.
    
    Parameters:
        data (pd.DataFrame): Weather data
        date_column (str): Name of the date column
        rainfall_column (str): Name of the rainfall column
        chart_type (str): Type of chart (line, bar, area)
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort data by date
    data = data.sort_values(by=date_column)
    
    # Plot based on chart type
    if chart_type == 'line':
        ax.plot(data[date_column], data[rainfall_column], marker='', linestyle='-', linewidth=2, color='green')
    elif chart_type == 'bar':
        ax.bar(data[date_column], data[rainfall_column], color='green', alpha=0.7)
    elif chart_type == 'area':
        ax.fill_between(data[date_column], data[rainfall_column], color='green', alpha=0.5)
    
    # Add titles and labels
    ax.set_title('Rainfall Over Time', fontsize=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rainfall Amount', fontsize=12)
    
    # Improve visual appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.autofmt_xdate()  # Rotate date labels
    
    plt.tight_layout()
    return fig

def plot_weather_comparison(data, date_column, weather_columns, chart_type='line'):
    """
    Plot multiple weather parameters for comparison.
    
    Parameters:
        data (pd.DataFrame): Weather data
        date_column (str): Name of the date column
        weather_columns (dict): Dictionary mapping display names to column names
        chart_type (str): Type of chart (line, area)
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    fig, axes = plt.subplots(len(weather_columns), 1, figsize=(12, 4 * len(weather_columns)), sharex=True)
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort data by date
    data = data.sort_values(by=date_column)
    
    # If there's only one parameter, axes won't be an array
    if len(weather_columns) == 1:
        axes = [axes]
    
    # Colors for different parameters
    colors = ['red', 'blue', 'purple', 'green', 'orange', 'brown']
    
    # Plot each parameter
    for i, (param_name, column_name) in enumerate(weather_columns.items()):
        ax = axes[i]
        color = colors[i % len(colors)]
        
        # Plot based on chart type
        if chart_type == 'line':
            ax.plot(data[date_column], data[column_name], marker='', linestyle='-', linewidth=2, color=color)
        elif chart_type == 'area':
            ax.fill_between(data[date_column], data[column_name], color=color, alpha=0.5)
        elif chart_type == 'bar':
            ax.bar(data[date_column], data[column_name], color=color, alpha=0.7)
        
        # Add titles and labels
        ax.set_title(f'{param_name} Over Time', fontsize=12)
        ax.set_ylabel(param_name, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add common x-axis label
    axes[-1].set_xlabel('Date', fontsize=12)
    fig.autofmt_xdate()  # Rotate date labels
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(data, columns):
    """
    Plot a heatmap showing correlation between weather parameters.
    
    Parameters:
        data (pd.DataFrame): Weather data
        columns (list): List of column names to include in correlation
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    # Calculate correlation matrix
    corr_matrix = data[columns].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        linewidths=0.5, 
        ax=ax,
        vmin=-1, 
        vmax=1,
        fmt=".2f"
    )
    
    # Add titles
    ax.set_title('Correlation Between Weather Parameters', fontsize=15)

    plt.tight_layout()
    return fig
