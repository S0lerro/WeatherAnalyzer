import os
import pandas as pd
import numpy as np
import re

def detect_file_type(filename):
    """
    Detect the file type based on file extension.
    
    Parameters:
        filename (str): Name of the file
    
    Returns:
        str: Type of the file (csv, excel, json, or unknown)
    """
    extension = os.path.splitext(filename)[1].lower()
    
    if extension == '.csv':
        return "csv"
    elif extension in ['.xlsx', '.xls']:
        return "excel"
    elif extension == '.json':
        return "json"
    else:
        return "unknown"

def validate_weather_data(data):
    """
    Validate if the data has required columns for weather analysis.
    
    Parameters:
        data (pd.DataFrame): Data to validate
    
    Returns:
        tuple: (is_valid, message)
    """
    # Check if the dataframe is empty
    if data.empty:
        return False, "The uploaded file contains no data."
    
    # Check for date/time column
    date_column = None
    for col in data.columns:
        # Look for columns that might contain date information
        if any(term in col.lower() for term in ['date', 'time', 'dt', 'timestamp']):
            # Try to convert to datetime
            try:
                pd.to_datetime(data[col])
                date_column = col
                break
            except:
                pass
    
    if date_column is None:
        # Try to find any column that can be converted to datetime
        for col in data.columns:
            try:
                pd.to_datetime(data[col])
                date_column = col
                break
            except:
                pass
    
    if date_column is None:
        return False, "No valid date/time column found."
    
    # Check for at least one weather parameter column
    weather_pattern = re.compile(
        r'temp|humidity|pressure|rain|precip|wind|cloud|snow|sun|weather',
        re.IGNORECASE
    )
    
    weather_columns = [col for col in data.columns if weather_pattern.search(col)]
    
    if not weather_columns:
        return False, "No weather parameter columns found (temperature, humidity, pressure, rainfall, etc.)."
    
    # Check if there's enough data for analysis
    if len(data) < 10:
        return False, "The dataset is too small for meaningful analysis (less than 10 records)."
    
    return True, "Data validation successful."

def identify_weather_columns(data):
    """
    Automatically identify weather-related columns in the dataset.
    
    Parameters:
        data (pd.DataFrame): Weather data
    
    Returns:
        dict: Dictionary mapping column types to column names
    """
    # Initialize dictionary to store identified columns
    column_mapping = {
        'date': None,
        'temperature': None,
        'humidity': None,
        'pressure': None,
        'rainfall': None,
        'wind_speed': None,
        'wind_direction': None,
        'cloud_cover': None
    }
    
    # Define patterns for different weather parameters
    patterns = {
        'date': re.compile(r'date|time|dt|timestamp', re.IGNORECASE),
        'temperature': re.compile(r'temp|temperature', re.IGNORECASE),
        'humidity': re.compile(r'humid|humidity|moisture', re.IGNORECASE),
        'pressure': re.compile(r'press|pressure|barometric', re.IGNORECASE),
        'rainfall': re.compile(r'rain|rainfall|precip|precipitation', re.IGNORECASE),
        'wind_speed': re.compile(r'wind.*speed|speed.*wind', re.IGNORECASE),
        'wind_direction': re.compile(r'wind.*dir|direction.*wind', re.IGNORECASE),
        'cloud_cover': re.compile(r'cloud|cover|overcast', re.IGNORECASE)
    }
    
    # Check each column
    for col in data.columns:
        # Check date column first
        if patterns['date'].search(col) and column_mapping['date'] is None:
            try:
                # Make sure it can be converted to datetime
                pd.to_datetime(data[col])
                column_mapping['date'] = col
                continue  # Skip other checks for this column
            except:
                pass
        
        # Check other weather parameters
        for param, pattern in patterns.items():
            if param == 'date':
                continue  # Already handled date
                
            if pattern.search(col) and column_mapping[param] is None:
                column_mapping[param] = col
                break  # Once matched, skip other parameter checks
    
    # If date column still not found, try to find any column that can be datetime
    if column_mapping['date'] is None:
        for col in data.columns:
            try:
                pd.to_datetime(data[col])
                column_mapping['date'] = col
                break
            except:
                pass
    
    # Remove None values from the mapping
    return {k: v for k, v in column_mapping.items() if v is not None}

def find_correlation(data, column1, column2):
    """
    Calculate correlation between two columns in the dataset.
    
    Parameters:
        data (pd.DataFrame): Weather data
        column1 (str): First column name
        column2 (str): Second column name
    
    Returns:
        float: Correlation coefficient
    """
    # Check if columns exist
    if column1 not in data.columns or column2 not in data.columns:
        return None
    
    # Calculate correlation
    return data[column1].corr(data[column2])

def detect_seasonality(data, date_column, value_column, period=None):
    """
    Detect seasonality in time series data.
    
    Parameters:
        data (pd.DataFrame): Weather data
        date_column (str): Name of the date column
        value_column (str): Name of the value column
        period (int, optional): Period to check (e.g., 7 for weekly, 12 for monthly, 365 for yearly)
    
    Returns:
        tuple: (has_seasonality, correlation_strength, best_period)
    """
    from statsmodels.tsa.stattools import acf
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort by date
    data = data.sort_values(by=date_column)
    
    # Remove missing values
    values = data[value_column].dropna().values
    
    # If period is not specified, try to detect best period
    best_period = period
    best_corr = 0
    
    if period is None:
        periods_to_try = [7, 30, 365]  # Weekly, monthly, yearly
        
        # Calculate autocorrelation for different lags
        acf_values = acf(values, nlags=max(periods_to_try), fft=True)
        
        # Find best period from the ones we're checking
        for p in periods_to_try:
            if p < len(acf_values):
                if abs(acf_values[p]) > best_corr:
                    best_corr = abs(acf_values[p])
                    best_period = p
    else:
        # Calculate autocorrelation for the given period
        acf_values = acf(values, nlags=period, fft=True)
        best_corr = abs(acf_values[-1]) if len(acf_values) > 1 else 0
    
    # Determine if there's significant seasonality
    has_seasonality = best_corr > 0.2  # Arbitrary threshold
    
    return has_seasonality, best_corr, best_period
