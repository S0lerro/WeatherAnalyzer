import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import datetime

def extract_date_features(dates):
    """
    Extract useful features from dates for machine learning.
    
    Parameters:
        dates (pandas.Series): Series of datetime values
    
    Returns:
        pandas.DataFrame: DataFrame with extracted date features
    """
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)
    
    # Extract features
    features = pd.DataFrame({
        'day_of_year': dates.dt.dayofyear,
        'day_of_month': dates.dt.day,
        'day_of_week': dates.dt.dayofweek,
        'month': dates.dt.month,
        'year': dates.dt.year,
        'is_weekend': dates.dt.dayofweek.isin([5, 6]).astype(int),
        'quarter': dates.dt.quarter,
        # Add sine and cosine features to capture seasonality
        'sin_day': np.sin(2 * np.pi * dates.dt.dayofyear / 365),
        'cos_day': np.cos(2 * np.pi * dates.dt.dayofyear / 365),
        'sin_month': np.sin(2 * np.pi * dates.dt.month / 12),
        'cos_month': np.cos(2 * np.pi * dates.dt.month / 12)
    })
    
    return features

def train_temperature_model(data, date_column, temp_column):
    """
    Train a model to predict temperature.
    
    Parameters:
        data (pd.DataFrame): Weather data
        date_column (str): Name of the date column
        temp_column (str): Name of the temperature column
    
    Returns:
        sklearn.pipeline.Pipeline: Trained model
    """
    # Extract date features
    date_features = extract_date_features(data[date_column])
    
    # Prepare features and target
    X = date_features
    y = data[temp_column]
    
    # Create pipeline with preprocessing and model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    model.fit(X, y)
    
    return model

def train_humidity_model(data, date_column, humid_column):
    """
    Train a model to predict humidity.
    
    Parameters:
        data (pd.DataFrame): Weather data
        date_column (str): Name of the date column
        humid_column (str): Name of the humidity column
    
    Returns:
        sklearn.pipeline.Pipeline: Trained model
    """
    # Extract date features
    date_features = extract_date_features(data[date_column])
    
    # Prepare features and target
    X = date_features
    y = data[humid_column]
    
    # Create pipeline with preprocessing and model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    model.fit(X, y)
    
    return model

def train_pressure_model(data, date_column, pressure_column):
    """
    Train a model to predict atmospheric pressure.
    
    Parameters:
        data (pd.DataFrame): Weather data
        date_column (str): Name of the date column
        pressure_column (str): Name of the pressure column
    
    Returns:
        sklearn.pipeline.Pipeline: Trained model
    """
    # Extract date features
    date_features = extract_date_features(data[date_column])
    
    # Prepare features and target
    X = date_features
    y = data[pressure_column]
    
    # Create pipeline with preprocessing and model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    model.fit(X, y)
    
    return model

def train_rainfall_model(data, date_column, rainfall_column):
    """
    Train a model to predict rainfall.
    
    Parameters:
        data (pd.DataFrame): Weather data
        date_column (str): Name of the date column
        rainfall_column (str): Name of the rainfall column
    
    Returns:
        sklearn.pipeline.Pipeline: Trained model
    """
    # Extract date features
    date_features = extract_date_features(data[date_column])
    
    # Prepare features and target
    X = date_features
    y = data[rainfall_column]
    
    # Create pipeline with preprocessing and model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    model.fit(X, y)
    
    return model

def predict_weather(model, prediction_dates, weather_type):
    """
    Make weather predictions for future dates.
    
    Parameters:
        model (sklearn.pipeline.Pipeline): Trained model
        prediction_dates (list): List of dates for prediction
        weather_type (str): Type of weather parameter being predicted
    
    Returns:
        np.array: Predicted values
    """
    # Extract features from prediction dates
    prediction_features = extract_date_features(prediction_dates)
    
    # Make predictions
    predictions = model.predict(prediction_features)
    
    # Ensure predictions are reasonable based on weather type
    if weather_type == "humidity":
        # Humidity should be between 0 and 100
        predictions = np.clip(predictions, 0, 100)
    elif weather_type == "rainfall":
        # Rainfall should not be negative
        predictions = np.clip(predictions, 0, None)
    elif weather_type == "pressure":
        # Typical atmospheric pressure range (adjust as needed)
        predictions = np.clip(predictions, 800, 1200)
    
    return predictions
