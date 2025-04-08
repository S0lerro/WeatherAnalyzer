import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import custom modules
from data_processor import preprocess_data, clean_data, aggregate_data
from visualization import (
    plot_temperature_trend, 
    plot_humidity_trend, 
    plot_pressure_trend, 
    plot_rainfall_trend,
    plot_weather_comparison,
    plot_correlation_heatmap
)
from prediction import (
    train_temperature_model, 
    train_humidity_model, 
    train_pressure_model, 
    train_rainfall_model,
    predict_weather
)
from utils import detect_file_type, validate_weather_data

# Set page configuration
st.set_page_config(
    page_title="Weather Data Analyzer",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("Weather Data Processing and Visualization")
st.markdown("""
This application allows you to import, process, and visualize weather data.
It provides statistical analysis and basic prediction capabilities.
""")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'date_column' not in st.session_state:
    st.session_state.date_column = None
if 'temp_column' not in st.session_state:
    st.session_state.temp_column = None
if 'humid_column' not in st.session_state:
    st.session_state.humid_column = None
if 'pressure_column' not in st.session_state:
    st.session_state.pressure_column = None
if 'rainfall_column' not in st.session_state:
    st.session_state.rainfall_column = None

# Sidebar for data input and configuration
with st.sidebar:
    st.header("Data Import")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload weather data file (CSV, Excel, JSON)", 
        type=["csv", "xlsx", "xls", "json"]
    )
    
    # Sample selection (if no file is uploaded)
    if uploaded_file is None:
        st.info("Please upload a weather data file to begin.")
    
    # Process the uploaded file
    if uploaded_file is not None:
        try:
            # Detect file type and read data
            file_type = detect_file_type(uploaded_file.name)
            
            if file_type == "csv":
                data = pd.read_csv(uploaded_file)
            elif file_type == "excel":
                data = pd.read_excel(uploaded_file)
            elif file_type == "json":
                data = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format.")
                data = None
            
            if data is not None:
                # Validate data
                is_valid, message = validate_weather_data(data)
                if is_valid:
                    st.success("Data loaded successfully!")
                    st.session_state.data = data
                else:
                    st.error(f"Invalid data format: {message}")
                    st.session_state.data = None
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.session_state.data = None
    
    # Column mapping (if data is loaded)
    if st.session_state.data is not None:
        st.header("Column Mapping")
        st.info("Please map your data columns to weather parameters")
        
        columns = st.session_state.data.columns.tolist()
        
        # Map date/time column
        date_col = st.selectbox("Date/Time Column", columns, index=0 if columns else None)
        st.session_state.date_column = date_col
        
        # Map weather parameter columns
        temp_col = st.selectbox("Temperature Column", ["None"] + columns, index=0)
        st.session_state.temp_column = temp_col if temp_col != "None" else None
        
        humid_col = st.selectbox("Humidity Column", ["None"] + columns, index=0)
        st.session_state.humid_column = humid_col if humid_col != "None" else None
        
        pressure_col = st.selectbox("Pressure Column", ["None"] + columns, index=0)
        st.session_state.pressure_column = pressure_col if pressure_col != "None" else None
        
        rainfall_col = st.selectbox("Rainfall Column", ["None"] + columns, index=0)
        st.session_state.rainfall_column = rainfall_col if rainfall_col != "None" else None
        
        # Data preprocessing options
        st.header("Data Preprocessing")
        handle_missing = st.checkbox("Handle missing values", value=True)
        remove_outliers = st.checkbox("Remove outliers", value=True)
        
        # Time aggregation options
        st.header("Time Aggregation")
        aggregation = st.selectbox(
            "Aggregate data by",
            ["None", "Hour", "Day", "Week", "Month"],
            index=0
        )
        
        # Apply preprocessing if button is clicked
        if st.button("Preprocess Data"):
            if st.session_state.date_column:
                processed_data = preprocess_data(
                    st.session_state.data,
                    st.session_state.date_column,
                    handle_missing=handle_missing,
                    remove_outliers=remove_outliers
                )
                
                if aggregation != "None":
                    processed_data = aggregate_data(
                        processed_data,
                        st.session_state.date_column,
                        aggregation.lower()
                    )
                
                st.session_state.data = processed_data
                st.success("Data preprocessing completed!")
            else:
                st.error("Please select a valid date column")
    
    # Prediction options (only shown if data is loaded and mapped)
    if (st.session_state.data is not None and 
        st.session_state.date_column is not None and
        (st.session_state.temp_column is not None or
         st.session_state.humid_column is not None or
         st.session_state.pressure_column is not None or
         st.session_state.rainfall_column is not None)):
        
        st.header("Prediction")
        prediction_days = st.slider("Days to predict", 1, 30, 7)
        
        if st.button("Generate Predictions"):
            data = st.session_state.data
            
            # Dictionary to store predictions
            predictions = {}
            
            # Train models and generate predictions for each weather parameter
            date_col = st.session_state.date_column
            last_date = pd.to_datetime(data[date_col].iloc[-1])
            
            # Generate prediction dates
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
            predictions["Date"] = prediction_dates
            
            # Temperature predictions
            if st.session_state.temp_column:
                model = train_temperature_model(
                    data, 
                    st.session_state.date_column, 
                    st.session_state.temp_column
                )
                temp_predictions = predict_weather(model, prediction_dates, "temperature")
                predictions["Temperature"] = temp_predictions
            
            # Humidity predictions
            if st.session_state.humid_column:
                model = train_humidity_model(
                    data, 
                    st.session_state.date_column, 
                    st.session_state.humid_column
                )
                humid_predictions = predict_weather(model, prediction_dates, "humidity")
                predictions["Humidity"] = humid_predictions
            
            # Pressure predictions
            if st.session_state.pressure_column:
                model = train_pressure_model(
                    data, 
                    st.session_state.date_column, 
                    st.session_state.pressure_column
                )
                pressure_predictions = predict_weather(model, prediction_dates, "pressure")
                predictions["Pressure"] = pressure_predictions
            
            # Rainfall predictions
            if st.session_state.rainfall_column:
                model = train_rainfall_model(
                    data, 
                    st.session_state.date_column, 
                    st.session_state.rainfall_column
                )
                rainfall_predictions = predict_weather(model, prediction_dates, "rainfall")
                predictions["Rainfall"] = rainfall_predictions
            
            # Save predictions to session state
            st.session_state.predictions = pd.DataFrame(predictions)
            st.success("Predictions generated successfully!")

# Main content area
if st.session_state.data is None:
    st.info("Please upload data using the sidebar options to begin analysis.")
else:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Overview", 
        "Visualization", 
        "Statistical Analysis",
        "Predictions"
    ])
    
    # Data Overview Tab
    with tab1:
        st.header("Data Overview")
        
        # Show basic info about the dataset
        st.subheader("Dataset Information")
        data_info = {
            "Number of Records": len(st.session_state.data),
            "Time Range": f"{st.session_state.data[st.session_state.date_column].min()} to {st.session_state.data[st.session_state.date_column].max()}",
            "Columns": ", ".join(st.session_state.data.columns)
        }
        st.json(data_info)
        
        # Show first few rows of the dataset
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(10))
        
        # Basic statistics for numerical columns
        st.subheader("Statistical Summary")
        st.dataframe(st.session_state.data.describe())
        
        # Missing values information
        st.subheader("Missing Values")
        missing_values = st.session_state.data.isnull().sum()
        st.dataframe(pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values': missing_values.values,
            'Percentage': (missing_values.values / len(st.session_state.data) * 100).round(2)
        }))
        
        # Download processed data option
        st.subheader("Download Processed Data")
        
        csv = st.session_state.data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_weather_data.csv",
            mime="text/csv"
        )
    
    # Visualization Tab
    with tab2:
        st.header("Data Visualization")
        
        # Time range filter
        st.subheader("Filter Time Range")
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(st.session_state.data[st.session_state.date_column]):
            st.session_state.data[st.session_state.date_column] = pd.to_datetime(
                st.session_state.data[st.session_state.date_column]
            )
        
        # Get min and max dates
        min_date = st.session_state.data[st.session_state.date_column].min().date()
        max_date = st.session_state.data[st.session_state.date_column].max().date()
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Filter data based on selected date range
        filtered_data = st.session_state.data[
            (st.session_state.data[st.session_state.date_column].dt.date >= start_date) & 
            (st.session_state.data[st.session_state.date_column].dt.date <= end_date)
        ]
        
        # Display charts
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Line Chart", "Area Chart", "Bar Chart"]
        )
        
        # Plot options
        plot_options = []
        if st.session_state.temp_column:
            plot_options.append("Temperature")
        if st.session_state.humid_column:
            plot_options.append("Humidity")
        if st.session_state.pressure_column:
            plot_options.append("Pressure")
        if st.session_state.rainfall_column:
            plot_options.append("Rainfall")
        plot_options.append("All Parameters")
        plot_options.append("Correlation Heatmap")
        
        selected_plot = st.selectbox("Select Plot", plot_options)
        
        # Display selected visualization
        if selected_plot == "Temperature" and st.session_state.temp_column:
            fig = plot_temperature_trend(
                filtered_data, 
                st.session_state.date_column, 
                st.session_state.temp_column,
                chart_type.lower().replace(" ", "_")
            )
            st.pyplot(fig)
            
        elif selected_plot == "Humidity" and st.session_state.humid_column:
            fig = plot_humidity_trend(
                filtered_data, 
                st.session_state.date_column, 
                st.session_state.humid_column,
                chart_type.lower().replace(" ", "_")
            )
            st.pyplot(fig)
            
        elif selected_plot == "Pressure" and st.session_state.pressure_column:
            fig = plot_pressure_trend(
                filtered_data, 
                st.session_state.date_column, 
                st.session_state.pressure_column,
                chart_type.lower().replace(" ", "_")
            )
            st.pyplot(fig)
            
        elif selected_plot == "Rainfall" and st.session_state.rainfall_column:
            fig = plot_rainfall_trend(
                filtered_data, 
                st.session_state.date_column, 
                st.session_state.rainfall_column,
                chart_type.lower().replace(" ", "_")
            )
            st.pyplot(fig)
            
        elif selected_plot == "All Parameters":
            columns_to_plot = {}
            if st.session_state.temp_column:
                columns_to_plot["Temperature"] = st.session_state.temp_column
            if st.session_state.humid_column:
                columns_to_plot["Humidity"] = st.session_state.humid_column
            if st.session_state.pressure_column:
                columns_to_plot["Pressure"] = st.session_state.pressure_column
            if st.session_state.rainfall_column:
                columns_to_plot["Rainfall"] = st.session_state.rainfall_column
                
            fig = plot_weather_comparison(
                filtered_data,
                st.session_state.date_column,
                columns_to_plot,
                chart_type.lower().replace(" ", "_")
            )
            st.pyplot(fig)
            
        elif selected_plot == "Correlation Heatmap":
            # Create a list of columns to include in correlation
            corr_columns = []
            
            if st.session_state.temp_column:
                corr_columns.append(st.session_state.temp_column)
            if st.session_state.humid_column:
                corr_columns.append(st.session_state.humid_column)
            if st.session_state.pressure_column:
                corr_columns.append(st.session_state.pressure_column)
            if st.session_state.rainfall_column:
                corr_columns.append(st.session_state.rainfall_column)
                
            if len(corr_columns) > 1:
                fig = plot_correlation_heatmap(filtered_data, corr_columns)
                st.pyplot(fig)
            else:
                st.warning("At least two weather parameters are needed for a correlation heatmap.")
    
    # Statistical Analysis Tab
    with tab3:
        st.header("Statistical Analysis")
        
        # Select parameters for analysis
        analysis_params = []
        
        if st.session_state.temp_column:
            analysis_params.append(st.session_state.temp_column)
        if st.session_state.humid_column:
            analysis_params.append(st.session_state.humid_column)
        if st.session_state.pressure_column:
            analysis_params.append(st.session_state.pressure_column)
        if st.session_state.rainfall_column:
            analysis_params.append(st.session_state.rainfall_column)
            
        selected_param = st.selectbox("Select parameter for analysis", analysis_params)
        
        if selected_param:
            # Time series decomposition
            st.subheader("Time Series Decomposition")
            
            # Check if we have enough data for decomposition (need at least 2 periods)
            try:
                # Convert date column to datetime
                filtered_data[st.session_state.date_column] = pd.to_datetime(filtered_data[st.session_state.date_column])
                
                # Set the date as index for time series analysis
                ts_data = filtered_data.set_index(st.session_state.date_column)[selected_param]
                
                # Check for frequency
                if not pd.infer_freq(ts_data.index):
                    st.warning("Data points are not evenly spaced. Using daily frequency for analysis.")
                    # Resample to daily frequency
                    ts_data = ts_data.resample('D').mean()
                    
                # Import the necessary libraries for decomposition
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Perform decomposition
                decomposition = seasonal_decompose(ts_data.dropna(), model='additive', period=7)  # Assume weekly seasonality
                
                # Plot decomposition
                fig, axs = plt.subplots(4, 1, figsize=(10, 12))
                decomposition.observed.plot(ax=axs[0])
                axs[0].set_title('Observed')
                decomposition.trend.plot(ax=axs[1])
                axs[1].set_title('Trend')
                decomposition.seasonal.plot(ax=axs[2])
                axs[2].set_title('Seasonality')
                decomposition.resid.plot(ax=axs[3])
                axs[3].set_title('Residuals')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Monthly and daily statistics
                st.subheader("Monthly Statistics")
                
                # Group by month
                monthly_data = filtered_data.copy()
                monthly_data['Month'] = pd.to_datetime(monthly_data[st.session_state.date_column]).dt.month
                monthly_stats = monthly_data.groupby('Month')[selected_param].agg(['mean', 'min', 'max', 'std'])
                monthly_stats.index = monthly_stats.index.map(lambda x: datetime(2022, x, 1).strftime('%B'))
                
                st.dataframe(monthly_stats)
                
                # Group by day of week
                st.subheader("Day of Week Statistics")
                
                daily_data = filtered_data.copy()
                daily_data['DayOfWeek'] = pd.to_datetime(daily_data[st.session_state.date_column]).dt.dayofweek
                daily_stats = daily_data.groupby('DayOfWeek')[selected_param].agg(['mean', 'min', 'max', 'std'])
                daily_stats.index = daily_stats.index.map(lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
                
                st.dataframe(daily_stats)
                
                # Rolling averages
                st.subheader("Rolling Averages")
                
                window_size = st.slider("Window size (days)", 2, 30, 7)
                
                rolling_mean = ts_data.rolling(window=window_size).mean()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ts_data.plot(ax=ax, alpha=0.5, label='Original')
                rolling_mean.plot(ax=ax, linewidth=2, label=f'{window_size}-day Moving Average')
                plt.legend()
                plt.title(f'Rolling Average for {selected_param}')
                st.pyplot(fig)
                
                # Statistical tests
                st.subheader("Statistical Tests")
                
                # Autocorrelation
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                
                st.write("Autocorrelation Function (ACF)")
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_acf(ts_data.dropna(), ax=ax)
                st.pyplot(fig)
                
                st.write("Partial Autocorrelation Function (PACF)")
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_pacf(ts_data.dropna(), ax=ax)
                st.pyplot(fig)
                
                # Distribution analysis
                st.subheader("Distribution Analysis")
                
                fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                
                # Histogram
                axs[0].hist(filtered_data[selected_param].dropna(), bins=20, alpha=0.7)
                axs[0].set_title(f'Histogram of {selected_param}')
                
                # Q-Q Plot
                from scipy import stats
                
                stats.probplot(filtered_data[selected_param].dropna(), dist="norm", plot=axs[1])
                axs[1].set_title('Q-Q Plot')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Descriptive statistics
                st.subheader("Descriptive Statistics")
                
                desc_stats = filtered_data[selected_param].describe()
                st.dataframe(pd.DataFrame(desc_stats).T)
                
                # Additional statistics
                skewness = filtered_data[selected_param].skew()
                kurtosis = filtered_data[selected_param].kurtosis()
                
                st.write(f"Skewness: {skewness:.4f}")
                st.write(f"Kurtosis: {kurtosis:.4f}")
                
            except Exception as e:
                st.error(f"Error in statistical analysis: {str(e)}")
                st.info("Make sure you have enough data for the analysis and that the time series is correctly formatted.")
    
    # Predictions Tab
    with tab4:
        st.header("Weather Predictions")
        
        if st.session_state.predictions is not None:
            # Display prediction results
            st.subheader("Predicted Weather Data")
            st.dataframe(st.session_state.predictions)
            
            # Visualization of predictions
            st.subheader("Prediction Visualization")
            
            # Available columns for visualization
            pred_columns = [col for col in st.session_state.predictions.columns if col != "Date"]
            
            if pred_columns:
                selected_pred_param = st.selectbox("Select parameter to visualize", pred_columns)
                
                # Plot predictions against actual data
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Determine which actual column corresponds to the selected prediction
                actual_col = None
                if selected_pred_param == "Temperature" and st.session_state.temp_column:
                    actual_col = st.session_state.temp_column
                elif selected_pred_param == "Humidity" and st.session_state.humid_column:
                    actual_col = st.session_state.humid_column
                elif selected_pred_param == "Pressure" and st.session_state.pressure_column:
                    actual_col = st.session_state.pressure_column
                elif selected_pred_param == "Rainfall" and st.session_state.rainfall_column:
                    actual_col = st.session_state.rainfall_column
                
                if actual_col:
                    # Plot historical data
                    ax.plot(
                        st.session_state.data[st.session_state.date_column],
                        st.session_state.data[actual_col],
                        label="Historical",
                        color="blue"
                    )
                    
                    # Plot predictions
                    ax.plot(
                        st.session_state.predictions["Date"],
                        st.session_state.predictions[selected_pred_param],
                        label="Predicted",
                        color="red",
                        marker="o"
                    )
                    
                    # Add shading for prediction uncertainty
                    # (using a simple approach of +/- 10% for illustration)
                    predictions = st.session_state.predictions[selected_pred_param].values
                    dates = st.session_state.predictions["Date"]
                    
                    ax.fill_between(
                        dates,
                        predictions * 0.9,  # Lower bound
                        predictions * 1.1,  # Upper bound
                        color="red",
                        alpha=0.2,
                        label="Prediction Interval"
                    )
                    
                    ax.set_title(f"Historical and Predicted {selected_pred_param}")
                    ax.set_xlabel("Date")
                    ax.set_ylabel(selected_pred_param)
                    ax.legend()
                    
                    st.pyplot(fig)
                else:
                    st.warning("Cannot find corresponding historical data for the selected prediction.")
                
                # Download predictions
                csv_predictions = st.session_state.predictions.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_predictions,
                    file_name="weather_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No prediction columns available for visualization.")
        else:
            st.info("Please generate predictions using the sidebar options.")
