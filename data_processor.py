import pandas as pd
import numpy as np
from scipy import stats

def preprocess_data(data, date_column, handle_missing=True, remove_outliers=True):
    """
    Preprocess weather data by handling missing values and outliers.
    
    Parameters:
        data (pd.DataFrame): Input weather data
        date_column (str): Name of the date/time column
        handle_missing (bool): Whether to handle missing values
        remove_outliers (bool): Whether to remove outliers
    
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Create a copy of the data to avoid modifying the original
    processed_data = data.copy()
    
    # Convert date column to datetime
    try:
        processed_data[date_column] = pd.to_datetime(processed_data[date_column])
    except Exception as e:
        print(f"Error converting date column: {str(e)}")
        
    # Sort data by date
    processed_data = processed_data.sort_values(by=date_column)
    
    # Handle missing values if requested
    if handle_missing:
        processed_data = clean_data(processed_data)
    
    # Remove outliers if requested
    if remove_outliers:
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in processed_data.columns:  # Make sure column exists
                # Calculate z-scores
                z_scores = stats.zscore(processed_data[column], nan_policy='omit')
                
                # Create a mask for values within 3 standard deviations
                mask = np.abs(z_scores) < 3
                
                # Replace outliers with NaN and then interpolate
                processed_data.loc[~mask, column] = np.nan
                processed_data[column] = processed_data[column].interpolate(method='linear')
    
    return processed_data

def clean_data(data):
    """
    Clean weather data by handling missing values.
    
    Parameters:
        data (pd.DataFrame): Input weather data
    
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Create a copy of the data
    cleaned_data = data.copy()
    
    # Get numeric columns
    numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
    
    # Handle missing values in numeric columns using interpolation
    for column in numeric_columns:
        if cleaned_data[column].isnull().sum() > 0:
            # Use linear interpolation for missing values
            cleaned_data[column] = cleaned_data[column].interpolate(method='linear')
            
            # For any remaining NaN values at the beginning or end, use forward/backward fill
            cleaned_data[column] = cleaned_data[column].fillna(method='ffill')
            cleaned_data[column] = cleaned_data[column].fillna(method='bfill')
    
    # Handle missing values in non-numeric columns using forward fill
    non_numeric_columns = cleaned_data.select_dtypes(exclude=[np.number]).columns
    
    for column in non_numeric_columns:
        if column in cleaned_data.columns and cleaned_data[column].isnull().sum() > 0:
            cleaned_data[column] = cleaned_data[column].fillna(method='ffill')
            cleaned_data[column] = cleaned_data[column].fillna(method='bfill')
    
    return cleaned_data

def aggregate_data(data, date_column, frequency):
    """
    Aggregate weather data based on a time frequency.
    
    Parameters:
        data (pd.DataFrame): Input weather data
        date_column (str): Name of the date/time column
        frequency (str): Aggregation frequency (hour, day, week, month)
    
    Returns:
        pd.DataFrame: Aggregated data
    """
    # Create a copy of the data
    aggregated_data = data.copy()
    
    # Make sure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(aggregated_data[date_column]):
        aggregated_data[date_column] = pd.to_datetime(aggregated_data[date_column])
    
    # Set date column as index
    aggregated_data = aggregated_data.set_index(date_column)
    
    # Determine the resampling frequency
    freq_map = {
        'hour': 'H',
        'day': 'D',
        'week': 'W',
        'month': 'M'
    }
    
    resample_freq = freq_map.get(frequency.lower(), 'D')  # Default to daily if not recognized
    
    # Aggregate numeric data (mean for most metrics, sum for rainfall/precipitation)
    numeric_columns = aggregated_data.select_dtypes(include=[np.number]).columns
    
    # Create a dictionary for aggregation methods
    agg_dict = {}
    
    for column in numeric_columns:
        # Use sum for rainfall/precipitation columns, mean for others
        if any(term in column.lower() for term in ['rain', 'precip', 'rainfall', 'precipitation']):
            agg_dict[column] = 'sum'
        else:
            agg_dict[column] = 'mean'
    
    # Perform resampling with appropriate aggregation
    if agg_dict:
        aggregated_data = aggregated_data.resample(resample_freq).agg(agg_dict)
    else:
        aggregated_data = aggregated_data.resample(resample_freq).mean()
    
    # Reset index to convert date back to a column
    aggregated_data = aggregated_data.reset_index()
    
    return aggregated_data
