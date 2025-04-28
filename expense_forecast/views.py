from django.shortcuts import render
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from django.utils.timezone import now
from expenses.models import Expense
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import os
import warnings
import datetime
warnings.filterwarnings("ignore")

# Try to import pmdarima and Prophet, but handle if they're not available
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError):
    PMDARIMA_AVAILABLE = False
    # Define a simple function for when pmdarima is not available
    def auto_arima(y, **kwargs):
        return None

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


# Fetch the data from the Expense model and create the forecast
@login_required(login_url='/authentication/login')
def forecast(request):
    # Fetch all expenses for the current user (for better trend analysis)
    expenses = Expense.objects.filter(owner=request.user).order_by('date')
    
    # Check if we have enough expenses for forecasting
    if len(expenses) < 10:
        # Try to use the ARIMA dataset as a baseline if no user expenses
        try:
            arima_data = pd.read_csv('arima_dataset.csv')
            data = pd.DataFrame({
                'Date': pd.to_datetime(arima_data['date']),
                'Expenses': arima_data['amount'],
                'Category': arima_data['category']
            })
            messages.success(request, "Using historical data for forecast baseline. Add more of your own expenses for personalized forecasts.")
        except Exception as e:
            messages.error(request, f"Not enough expenses to make a forecast. Please add more expenses (minimum 10). Error: {str(e)}")
            return render(request, 'expense_forecast/index.html')
    else:
        # Create a DataFrame from the user's expenses
        data = pd.DataFrame({
            'Date': [expense.date for expense in expenses], 
            'Expenses': [expense.amount for expense in expenses], 
            'Category': [expense.category for expense in expenses]
        })
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Determine the forecast approach based on data size
    if len(data) >= 30:
        context = advanced_forecast(data, request.user.username)
    else:
        context = simple_forecast(data, request.user.username)
    
    return render(request, 'expense_forecast/index.html', context)


def preprocess_data(data):
    """Preprocess expense data for better forecasting"""
    # Set date as index and sort
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    
    # Remove extreme outliers (expenses beyond 3 standard deviations from mean)
    mean_expense = data['Expenses'].mean()
    std_expense = data['Expenses'].std()
    
    if std_expense > 0:  # Only filter if we have variation in the data
        lower_bound = max(0, mean_expense - 3 * std_expense)  # Ensure we don't go below 0
        upper_bound = mean_expense + 3 * std_expense
        data = data[(data['Expenses'] >= lower_bound) & (data['Expenses'] <= upper_bound)]
    
    # Resample to daily frequency and handle missing values
    data.set_index('Date', inplace=True)
    if len(data) > 14:  # Only resample if we have enough data
        # First, aggregate expenses by day (sum if multiple expenses per day)
        daily_data = data['Expenses'].resample('D').sum()
        
        # Fill missing days using linear interpolation first, then forward/backward fill
        daily_data = daily_data.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        
        # If still have NaNs, fill with the mean
        daily_data = daily_data.fillna(daily_data.mean())
        
        # Create a new DataFrame with the resampled data and maintain Category if needed
        if 'Category' in data.columns:
            # Most common category per day for categorization
            category_data = data.reset_index()
            category_counts = category_data.groupby([category_data['Date'].dt.date, 'Category']).size()
            most_common_categories = category_counts.groupby(level=0).idxmax()
            category_map = {date: cat[1] if isinstance(cat, tuple) else 'Other' 
                         for date, cat in most_common_categories.items()}
            
            # Create new dataframe with daily data and mapped categories
            dates = daily_data.index
            categories = [category_map.get(date.date(), 'Other') for date in dates]
            data = pd.DataFrame({'Expenses': daily_data, 'Category': categories}, index=dates)
        else:
            data = pd.DataFrame({'Expenses': daily_data})
    
    return data


def detect_and_handle_anomalies(data):
    """
    Enhanced anomaly detection with multiple techniques to identify outliers
    """
    try:
        # Create a copy of the data to avoid modifying the original
        cleaned_data = data.copy()
        
        # Method 1: Z-score with rolling window
        rolling_mean = data['Expenses'].rolling(window=7, min_periods=1).mean()
        rolling_std = data['Expenses'].rolling(window=7, min_periods=1).std()
        rolling_std = rolling_std.replace(0, data['Expenses'].std() or 1)  # Avoid division by zero
        
        z_scores = (data['Expenses'] - rolling_mean) / rolling_std
        
        # Method 2: IQR (Interquartile Range) method
        q1 = data['Expenses'].quantile(0.25)
        q3 = data['Expenses'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Method 3: Local Outlier Factor for more advanced detection if sklearn is available
        try:
            from sklearn.neighbors import LocalOutlierFactor
            if len(data) >= 10:  # Need minimum number of points
                # Reshape data for LOF
                X = data['Expenses'].values.reshape(-1, 1)
                
                # Apply LOF
                lof = LocalOutlierFactor(n_neighbors=min(5, len(data)-1), contamination=0.1)
                outlier_scores = lof.fit_predict(X)
                
                # Mark LOF outliers (returns -1 for outliers)
                lof_anomalies = data.index[outlier_scores == -1]
                
                print(f"LOF found {len(lof_anomalies)} anomalies")
            else:
                lof_anomalies = []
        except Exception as e:
            print(f"Could not apply LOF: {e}")
            lof_anomalies = []
        
        # Combine detection methods to identify anomalies
        z_score_anomalies = data.index[abs(z_scores) > 3]
        iqr_anomalies = data.index[(data['Expenses'] < lower_bound) | (data['Expenses'] > upper_bound)]
        
        # Get unique anomalies from all methods
        all_anomalies = set(z_score_anomalies) | set(iqr_anomalies) | set(lof_anomalies)
        print(f"Found {len(all_anomalies)} total anomalies using combined methods")
        
        # Handle anomalies with more intelligent replacement
        for idx in all_anomalies:
            # Get nearby non-anomaly values within a 14-day window
            window_start = idx - pd.Timedelta(days=14)
            window_end = idx + pd.Timedelta(days=14)
            
            nearby = data.loc[(data.index >= window_start) & (data.index <= window_end)]
            nearby = nearby[~nearby.index.isin(all_anomalies)]
            
            if len(nearby) > 0:
                # Use median of nearby values for replacement
                cleaned_data.at[idx, 'Expenses'] = nearby['Expenses'].median()
            else:
                # If no nearby normal values, use overall median
                cleaned_data.at[idx, 'Expenses'] = data['Expenses'].median()
        
        # For extreme anomalies (over 5 std deviations), consider removing rather than replacing
        extreme_anomalies = data.index[abs(z_scores) > 5]
        if len(extreme_anomalies) > 0 and len(extreme_anomalies) < len(data) * 0.05:  # Only drop if less than 5% of data
            print(f"Dropping {len(extreme_anomalies)} extreme anomalies")
            cleaned_data = cleaned_data.drop(extreme_anomalies)
        
        return cleaned_data
    
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        # Return original data if anomaly detection fails
        return data


def add_features(data):
    """
    Enhanced feature engineering to capture more patterns in expense data
    """
    feature_data = data.copy()
    
    # Basic time features
    feature_data['day_of_week'] = feature_data.index.dayofweek
    feature_data['day_of_month'] = feature_data.index.day
    feature_data['month'] = feature_data.index.month
    feature_data['quarter'] = feature_data.index.quarter
    feature_data['year'] = feature_data.index.year
    feature_data['is_weekend'] = (feature_data.index.dayofweek >= 5).astype(int)
    
    # Beginning/end of month indicators (common for recurring expenses)
    feature_data['is_start_of_month'] = (feature_data.index.day <= 5).astype(int)
    feature_data['is_end_of_month'] = (feature_data.index.day >= 25).astype(int)
    
    # Middle of month indicator (common for mid-month expenses)
    feature_data['is_mid_month'] = ((feature_data.index.day >= 14) & 
                                   (feature_data.index.day <= 16)).astype(int)
    
    # Pay period indicators (common scenarios)
    feature_data['is_biweekly_payday'] = False  # Will set below
    
    # Try to detect bi-weekly payment patterns
    try:
        # Look for expense spikes that might indicate paydays
        if len(data) >= 28:  # At least 4 weeks of data
            expenses = data['Expenses']
            # Use rolling max with 3-day window to find expense spikes
            rolling_max = expenses.rolling(window=3, center=True).max()
            
            # Identify days with expenses at least 1.5x the median
            high_expense_days = expenses > (1.5 * expenses.median())
            
            # Look for biweekly patterns in high expense days
            for start_day in range(14):  # Try each possible starting day
                # Check if there's a pattern of high expenses every 14 days
                pattern_days = [start_day + i*14 for i in range(10)]
                pattern_days = [d for d in pattern_days if d < len(expenses)]
                
                # If most of these days have high expenses, we found a pattern
                if sum(high_expense_days.iloc[pattern_days]) / len(pattern_days) > 0.5:
                    # Set the biweekly payday feature
                    for day in pattern_days:
                        if day < len(feature_data):
                            feature_data.iloc[day, feature_data.columns.get_loc('is_biweekly_payday')] = True
                    break
    except Exception as e:
        print(f"Error detecting payment patterns: {e}")
    
    # Holiday detection
    try:
        from holidays import country_holidays
        
        # Try to detect the most relevant country based on data patterns
        potential_countries = ['US', 'GB', 'IN', 'CA', 'AU']
        best_country = None
        max_correlation = -1
        
        for country_code in potential_countries:
            try:
                # Get holidays for the date range in our data
                years = feature_data.index.year.unique()
                country_holiday_list = country_holidays(country_code, years=years)
                
                # Mark holidays in the data
                feature_data[f'is_holiday_{country_code}'] = [
                    date.date() in country_holiday_list for date in feature_data.index
                ]
                
                # Check correlation with expenses
                correlation = feature_data['Expenses'].corr(
                    feature_data[f'is_holiday_{country_code}'].astype(int)
                )
                
                if abs(correlation) > max_correlation:
                    max_correlation = abs(correlation)
                    best_country = country_code
            except Exception:
                continue
                
        # Keep only the most relevant country's holidays
        if best_country:
            feature_data['is_holiday'] = feature_data[f'is_holiday_{best_country}']
            print(f"Detected {best_country} as the most relevant holiday calendar")
            
            # Drop individual country columns
            for country_code in potential_countries:
                if f'is_holiday_{country_code}' in feature_data.columns:
                    feature_data = feature_data.drop(f'is_holiday_{country_code}', axis=1)
    except Exception as e:
        print(f"Could not add holiday information: {e}")
        # Add a default holiday column (all False)
        feature_data['is_holiday'] = False
    
    # Add month progress feature (0 at start of month, 1 at end)
    # This helps capture gradual changes throughout the month
    feature_data['month_progress'] = (feature_data.index.day - 1) / (
        feature_data.index.days_in_month - 1
    )
    
    # Spending velocity features (rate of change in expenses)
    try:
        # Rolling 3-day expense accumulation
        feature_data['rolling_3day_sum'] = feature_data['Expenses'].rolling(window=3, min_periods=1).sum()
        
        # Rolling 7-day expense accumulation
        feature_data['rolling_7day_sum'] = feature_data['Expenses'].rolling(window=7, min_periods=1).sum()
        
        # Expenses relative to typical day
        day_of_week_avg = feature_data.groupby('day_of_week')['Expenses'].transform('mean')
        feature_data['expense_vs_weekday_avg'] = feature_data['Expenses'] / day_of_week_avg
        
        # Fill NaN values
        feature_data = feature_data.fillna(0)
        
    except Exception as e:
        print(f"Error creating velocity features: {e}")
    
    # Add cyclical features for day of week, day of month, month
    # This preserves the cyclical nature of these features (e.g., Dec is close to Jan)
    try:
        # Day of week (cycle of 7 days)
        feature_data['day_of_week_sin'] = np.sin(2 * np.pi * feature_data['day_of_week'] / 7)
        feature_data['day_of_week_cos'] = np.cos(2 * np.pi * feature_data['day_of_week'] / 7)
        
        # Day of month (cycle depends on month length)
        days_in_month = feature_data.index.days_in_month
        day_of_month_norm = (feature_data['day_of_month'] - 1) / days_in_month
        feature_data['day_of_month_sin'] = np.sin(2 * np.pi * day_of_month_norm)
        feature_data['day_of_month_cos'] = np.cos(2 * np.pi * day_of_month_norm)
        
        # Month of year (cycle of 12 months)
        feature_data['month_sin'] = np.sin(2 * np.pi * (feature_data['month'] - 1) / 12)
        feature_data['month_cos'] = np.cos(2 * np.pi * (feature_data['month'] - 1) / 12)
        
    except Exception as e:
        print(f"Error creating cyclical features: {e}")
    
    # Create lag features (previous days' expenses)
    for lag in [1, 3, 7, 14]:
        if len(feature_data) > lag:
            feature_data[f'expense_lag_{lag}'] = feature_data['Expenses'].shift(lag)
    
    # Add categorical spending patterns if Category is in the data
    if 'Category' in feature_data.columns:
        try:
            # One-hot encode categories
            dummies = pd.get_dummies(feature_data['Category'], prefix='category')
            feature_data = pd.concat([feature_data, dummies], axis=1)
            
            # Create lagged category indicators
            # This helps the model understand category-specific temporal patterns
            for lag in [1, 7]:
                if len(feature_data) > lag:
                    for col in dummies.columns:
                        feature_data[f'{col}_lag_{lag}'] = dummies[col].shift(lag)
        except Exception as e:
            print(f"Error processing categories: {e}")
    
    # Fill NaN values in lag features
    feature_data = feature_data.fillna(0)
    
    return feature_data


def category_forecasts(data):
    """Calculate simple category forecasts based on historical proportions"""
    # Return an empty dict if no category data
    if 'Category' not in data.columns:
        return {'category_forecasts': {}}
    
    # Calculate total spent by category
    category_totals = data.groupby('Category')['Expenses'].sum()
    
    # Calculate proportion by category
    total_expenses = category_totals.sum()
    category_proportions = category_totals / total_expenses if total_expenses > 0 else 0
    
    # Default forecast
    return {'category_forecasts': {cat: round(float(prop), 2) for cat, prop in category_proportions.items()}}


def simple_forecast(data, username):
    """Simple forecasting for smaller datasets"""
    # Use a simple ARIMA model for forecasting
    try:
        # Fit a simple ARIMA model
        model = ARIMA(data['Expenses'], order=(1, 1, 1))
        model_fit = model.fit()
        
        # Forecast next 30 days
        forecast_steps = 30
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Create index for forecasted dates
        current_date = datetime.datetime.now().date()
        next_day = current_date + pd.DateOffset(days=1)
        forecast_index = pd.date_range(start=next_day, periods=forecast_steps, freq='D')
        
        # Create a DataFrame for the forecast
        forecast_data = pd.DataFrame({'Date': forecast_index, 'Forecasted_Expenses': forecast})
        
        # Get predicted vs actual values for accuracy calculation
        predictions = model_fit.predict()
        
        # Calculate metrics
        mae = mean_absolute_error(data['Expenses'].values[:len(predictions)], predictions)
        
        # Calculate MAPE safely
        with np.errstate(divide='ignore', invalid='ignore'):
            individual_mape = np.abs((data['Expenses'].values[:len(predictions)] - predictions) / data['Expenses'].values[:len(predictions)]) * 100
        
        # Handle infinity and NaN values
        individual_mape = individual_mape[~np.isinf(individual_mape) & ~np.isnan(individual_mape)]
        mape = np.mean(individual_mape) if len(individual_mape) > 0 else 100
        
        # Calculate accuracy
        accuracy = max(0, 100 - min(mape, 100))
        
        # Create a plot
        fig = go.Figure()
        
        # Actual expenses
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Expenses'],
            mode='lines+markers',
            name='Actual Expenses'
        ))
        
        # Forecasted expenses
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast,
            mode='lines+markers',
            name='Forecasted Expenses',
            line=dict(color='firebrick', dash='dot')
        ))
        
        fig.update_layout(
            title='Expense Forecast for Next 30 Days',
            xaxis_title='Date',
            yaxis_title='Expenses (₹)',
            template='plotly_white'
        )
        
        plot_div = plot(fig, output_type='div', include_plotlyjs=True)
        
        # Calculate total forecasted expenses
        total_forecasted_expenses = np.sum(forecast)
        
        # Forecast categories
        if 'Category' in data.columns:
            category_forecasts_dict = forecast_categories_advanced(data, forecast_index, forecast)
        else:
            category_forecasts_dict = category_forecasts(data)
        
        # Print metrics
        print(f"Model Evaluation for {username}'s Simple Expense Forecast:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Forecast Accuracy: {accuracy:.2f}%")
        
        return {
            'forecast_data': forecast_data.to_dict(orient='records'),
            'total_forecasted_expenses': round(total_forecasted_expenses, 2),
            'category_forecasts': category_forecasts_dict,
            'model_accuracy': round(accuracy, 2),
            'mean_absolute_error': round(mae, 2),
            'mape': round(mape, 2),
            'plot_div': plot_div,
            'model_name': 'simple_arima'
        }
    except Exception as e:
        # Fallback to simple mean-based forecast
        print(f"Error in simple forecast: {e}")
        mean_expense = data['Expenses'].mean()
        
        # Create forecast with just the mean repeated
        forecast_steps = 30
        forecast = np.array([mean_expense] * forecast_steps)
        
        # Create index for forecasted dates
        current_date = datetime.datetime.now().date()
        next_day = current_date + pd.DateOffset(days=1)
        forecast_index = pd.date_range(start=next_day, periods=forecast_steps, freq='D')
        
        # Create a DataFrame for the forecast
        forecast_data = pd.DataFrame({'Date': forecast_index, 'Forecasted_Expenses': forecast})
        
        # Simple figure
        fig = go.Figure()
        
        # Actual expenses
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Expenses'],
            mode='lines+markers',
            name='Actual Expenses'
        ))
        
        # Forecasted expenses
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast,
            mode='lines+markers',
            name='Forecasted Expenses (Mean)',
            line=dict(color='firebrick', dash='dot')
        ))
        
        fig.update_layout(
            title='Simple Mean-Based Expense Forecast',
            xaxis_title='Date',
            yaxis_title='Expenses (₹)',
            template='plotly_white'
        )
        
        plot_div = plot(fig, output_type='div', include_plotlyjs=True)
        
        # Calculate total forecasted expenses
        total_forecasted_expenses = np.sum(forecast)
        
        # Forecast categories using basic method
        category_forecasts_dict = category_forecasts(data)
        
        return {
            'forecast_data': forecast_data.to_dict(orient='records'),
            'total_forecasted_expenses': round(total_forecasted_expenses, 2),
            'category_forecasts': category_forecasts_dict,
            'model_accuracy': 0,  # Cannot calculate accuracy for mean forecast
            'mean_absolute_error': 0,
            'mape': 100,
            'plot_div': plot_div,
            'model_name': 'mean'
        }


def forecast_categories_advanced(data, forecast_dates, forecast_amount):
    """
    Advanced category forecast with multiple techniques to split forecasted expenses by category
    """
    try:
        # Check if we have category data
        if 'Category' not in data.columns:
            return category_forecasts(data)  # Fall back to basic method
        
        # Create DataFrame with total forecasted amount for each day
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted_Total': forecast_amount
        })
        forecast_df.set_index('Date', inplace=True)
        
        # Get the unique categories from historical data
        categories = data['Category'].unique()
        
        # Initialize the category forecasts DataFrame
        category_forecast = pd.DataFrame(index=forecast_df.index, columns=categories)
        
        # Method 1: Time-based category allocation (using patterns by day of week)
        try:
            # Calculate historical proportion by category and day of week
            data_with_dow = data.copy()
            data_with_dow['day_of_week'] = data_with_dow.index.dayofweek
            
            # Get proportion of spending by category per day of week
            category_dow_props = data_with_dow.groupby(['day_of_week', 'Category'])['Expenses'].sum()
            dow_totals = data_with_dow.groupby('day_of_week')['Expenses'].sum()
            
            # Create a dictionary of day of week -> category -> proportion
            dow_category_props = {}
            for dow in range(7):
                if dow in dow_totals.index and dow_totals[dow] > 0:
                    dow_category_props[dow] = {}
                    for cat in categories:
                        try:
                            dow_category_props[dow][cat] = category_dow_props[dow, cat] / dow_totals[dow]
                        except (KeyError, ZeroDivisionError):
                            dow_category_props[dow][cat] = 0
            
            # Apply these proportions to each forecast date based on its day of week
            for date, row in forecast_df.iterrows():
                dow = date.dayofweek
                total_forecast = row['Forecasted_Total']
                
                if dow in dow_category_props:
                    for cat in categories:
                        if cat in dow_category_props[dow]:
                            category_forecast.at[date, cat] = total_forecast * dow_category_props[dow][cat]
        except Exception as e:
            print(f"Error in time-based category allocation: {e}")
            
        # Method 2: Seasonal category allocation (using patterns by month)
        try:
            # Calculate historical proportion by category and month
            data_with_month = data.copy()
            data_with_month['month'] = data_with_month.index.month
            
            # Get proportion of spending by category per month
            category_month_props = data_with_month.groupby(['month', 'Category'])['Expenses'].sum()
            month_totals = data_with_month.groupby('month')['Expenses'].sum()
            
            # Create a dictionary of month -> category -> proportion
            month_category_props = {}
            for month in range(1, 13):
                if month in month_totals.index and month_totals[month] > 0:
                    month_category_props[month] = {}
                    for cat in categories:
                        try:
                            month_category_props[month][cat] = category_month_props[month, cat] / month_totals[month]
                        except (KeyError, ZeroDivisionError):
                            month_category_props[month][cat] = 0
            
            # Apply these proportions to each forecast date based on its month
            for date, row in forecast_df.iterrows():
                month = date.month
                total_forecast = row['Forecasted_Total']
                
                if month in month_category_props:
                    for cat in categories:
                        if cat in month_category_props[month]:
                            # Blend with previous method's results if they exist
                            if not pd.isna(category_forecast.at[date, cat]):
                                category_forecast.at[date, cat] = (
                                    category_forecast.at[date, cat] * 0.6 + 
                                    total_forecast * month_category_props[month][cat] * 0.4
                                )
                            else:
                                category_forecast.at[date, cat] = total_forecast * month_category_props[month][cat]
        except Exception as e:
            print(f"Error in seasonal category allocation: {e}")
        
        # Method 3: Recent trends (giving more weight to recent spending patterns)
        try:
            # Get data from the most recent month if available
            recent_cutoff = data.index.max() - pd.Timedelta(days=30)
            recent_data = data[data.index >= recent_cutoff]
            
            if len(recent_data) >= 14:  # Enough recent data to use
                # Calculate recent proportions
                recent_totals = recent_data.groupby('Category')['Expenses'].sum()
                recent_total = recent_totals.sum()
                
                if recent_total > 0:
                    recent_props = {cat: recent_totals.get(cat, 0) / recent_total for cat in categories}
                    
                    # Apply recent proportions with more weight
                    for date, row in forecast_df.iterrows():
                        total_forecast = row['Forecasted_Total']
                        for cat in categories:
                            if cat in recent_props:
                                # Blend with higher weight for recent patterns
                                if not pd.isna(category_forecast.at[date, cat]):
                                    category_forecast.at[date, cat] = (
                                        category_forecast.at[date, cat] * 0.3 + 
                                        total_forecast * recent_props[cat] * 0.7
                                    )
                                else:
                                    category_forecast.at[date, cat] = total_forecast * recent_props[cat]
        except Exception as e:
            print(f"Error in recent trends allocation: {e}")
        
        # Method 4: Recurring expenses (looking for regular patterns by category)
        try:
            # Detect recurring expenses for each category
            recurring_patterns = {}
            
            # Dict to track first occurrence day of month for each category
            first_occurrence = {}
            
            # Look for spending that happens on the same day each month in each category
            data_copy = data.copy()
            for cat in categories:
                cat_data = data_copy[data_copy['Category'] == cat]
                if len(cat_data) >= 3:  # Need enough data points
                    # Check for consistent day-of-month patterns
                    day_counts = cat_data.index.day.value_counts()
                    # If a day appears regularly (at least 2 times), consider it recurring
                    recurring_days = day_counts[day_counts >= 2].index
                    
                    if len(recurring_days) > 0:
                        recurring_patterns[cat] = recurring_days
                        # Get the first occurrence day for sorting priority
                        first_occurrence[cat] = recurring_days[0]
            
            # Apply recurring patterns - higher probability of expense on historically recurring days
            for date, row in forecast_df.iterrows():
                day = date.day
                total_forecast = row['Forecasted_Total']
                
                # Adjust category allocation based on recurring patterns
                for cat in recurring_patterns:
                    if day in recurring_patterns[cat]:
                        # Increase weight for this category on recurring days
                        if not pd.isna(category_forecast.at[date, cat]):
                            category_forecast.at[date, cat] *= 1.5  # 50% increase for recurring day
        except Exception as e:
            print(f"Error in recurring expenses allocation: {e}")
        
        # Handle edge cases and normalize
        # Ensure we account for 100% of forecasted expenses
        category_forecast.fillna(0, inplace=True)
        
        # Normalize the forecasts to match the total
        for date, row in forecast_df.iterrows():
            total_forecast = row['Forecasted_Total']
            category_sum = category_forecast.loc[date].sum()
            
            if category_sum > 0:
                # Normalize to match total forecast
                category_forecast.loc[date] = category_forecast.loc[date] * (total_forecast / category_sum)
            else:
                # If all categories are zero, distribute evenly
                for cat in categories:
                    category_forecast.at[date, cat] = total_forecast / len(categories)
        
        # Calculate total by category for the 30-day forecast period
        category_totals = category_forecast.sum()
        
        # Return as a dictionary
        return {
            'category_forecasts': {cat: round(float(category_totals[cat]), 2) for cat in categories},
            'forecast_by_date_category': category_forecast.reset_index().to_dict(orient='records')
        }
    
    except Exception as e:
        print(f"Error in advanced category forecast: {e}")
        return category_forecasts(data)  # Fall back to simple method


def advanced_forecast(data, username):
    """Advanced forecasting with multiple models for larger datasets"""
    results = {}
    best_accuracy = -1
    best_model = None
    
    # Enhance the data preprocessing for better accuracy
    data = detect_and_handle_anomalies(data)
    
    # Add additional features if we have enough data points
    if len(data) >= 14:
        data_with_features = add_features(data.copy())
    else:
        data_with_features = data.copy()
    
    # Try multiple modeling approaches
    
    # 1. ARIMA Model with optimized parameters
    try:
        # Try to determine better ARIMA parameters based on data
        from statsmodels.tsa.stattools import acf, pacf
        
        # Calculate ACF and PACF to help determine order
        acf_vals = acf(data['Expenses'], nlags=14)
        pacf_vals = pacf(data['Expenses'], nlags=14)
        
        # Choose p and q based on significant lags in PACF and ACF
        p = min(3, sum(np.abs(pacf_vals[1:5]) > 0.2))
        q = min(3, sum(np.abs(acf_vals[1:5]) > 0.2))
        d = 1  # Default differencing
        
        # Use more robust parameter combinations
        arima_orders = [(p, d, q), (1, 1, 1), (2, 1, 2), (0, 1, 2)]
        
        best_arima_aic = float('inf')
        best_arima_model = None
        best_arima_preds = None
        best_arima_order = None
        
        # Try different ARIMA parameter combinations
        for order in arima_orders:
            try:
                arima_model = ARIMA(data['Expenses'], order=order)
                arima_fit = arima_model.fit()
                
                # Calculate AIC and store if best model
                if arima_fit.aic < best_arima_aic:
                    best_arima_aic = arima_fit.aic
                    best_arima_model = arima_fit
                    best_arima_preds = arima_fit.predict()
                    best_arima_order = order
            except Exception as e:
                print(f"ARIMA order {order} failed: {e}")
                continue
        
        # Store the best ARIMA model
        if best_arima_model:
            results['arima'] = {
                'model': best_arima_model,
                'predictions': best_arima_preds,
                'order': best_arima_order
            }
            print(f"Best ARIMA model fitted successfully with order {best_arima_order}")
    except Exception as e:
        print(f"ARIMA error: {e}")
        # Fallback to default ARIMA
        try:
            arima_model = ARIMA(data['Expenses'], order=(2, 1, 2))
            arima_fit = arima_model.fit()
            arima_preds = arima_fit.predict()
            
            results['arima'] = {
                'model': arima_fit,
                'predictions': arima_preds,
                'order': (2, 1, 2)
            }
            print("Default ARIMA model fitted successfully")
        except Exception as e:
            print(f"Default ARIMA also failed: {e}")
    
    # 2. Auto ARIMA Model (if available)
    if PMDARIMA_AVAILABLE:
        try:
            # Use seasonal decomposition to inform auto_arima parameters
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomposition = seasonal_decompose(data['Expenses'], model='additive', period=7)
                seasonal_strength = np.std(decomposition.seasonal) / np.std(data['Expenses'] - decomposition.trend)
                use_seasonal = seasonal_strength > 0.1  # Use seasonal if seasonal component is strong enough
            except:
                use_seasonal = True  # Default to using seasonality
            
            # Configure auto_arima with more robust parameters
            auto_model = auto_arima(data['Expenses'], 
                                  start_p=0, start_q=0, max_p=5, max_q=5, max_d=2,
                                  seasonal=use_seasonal, m=7,  # Weekly seasonality
                                  stepwise=True, error_action='ignore', 
                                  information_criterion='aic',  # Use AIC for model selection
                                  suppress_warnings=True,
                                  n_fits=50,  # Increase number of models to try
                                  trace=True)  # More feedback
            
            auto_model_fit = auto_model.fit(data['Expenses'])
            auto_preds = auto_model_fit.predict_in_sample()
            
            results['auto_arima'] = {
                'model': auto_model,
                'predictions': auto_preds,
                'order': auto_model.order
            }
            
            print(f"Auto ARIMA order: {auto_model.order}")
        except Exception as e:
            print(f"Auto ARIMA error: {e}")
    
    # 3. Seasonal ARIMA Model with optimal seasonality detection
    try:
        # Try to determine optimal seasonality period from data
        # For expenses, this could be weekly (7), bi-weekly (14), or monthly (30)
        potential_seasons = [7, 14, 30]
        best_sarimax_aic = float('inf')
        best_sarimax_model = None
        best_sarimax_preds = None
        best_sarimax_config = None
        
        for season in potential_seasons:
            # Only use seasons that make sense with the data length
            if len(data) >= season * 2:
                try:
                    sarimax_model = SARIMAX(data['Expenses'], 
                                          order=(1, 1, 1), 
                                          seasonal_order=(1, 0, 1, season),
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
                    sarimax_fit = sarimax_model.fit(disp=False, maxiter=200)
                    
                    if sarimax_fit.aic < best_sarimax_aic:
                        best_sarimax_aic = sarimax_fit.aic
                        best_sarimax_model = sarimax_fit
                        best_sarimax_preds = sarimax_fit.predict()
                        best_sarimax_config = season
                except Exception as e:
                    print(f"SARIMAX with season {season} failed: {e}")
                    continue
        
        # Store best SARIMAX model
        if best_sarimax_model:
            results['sarimax'] = {
                'model': best_sarimax_model,
                'predictions': best_sarimax_preds,
                'season': best_sarimax_config
            }
            print(f"Best SARIMAX model fitted successfully with season {best_sarimax_config}")
        else:
            # Fallback to default SARIMAX
            sarimax_model = SARIMAX(data['Expenses'], 
                                   order=(1, 1, 1), 
                                   seasonal_order=(1, 0, 1, 7))
            sarimax_fit = sarimax_model.fit(disp=False, maxiter=200)
            sarimax_preds = sarimax_fit.predict()
            
            results['sarimax'] = {
                'model': sarimax_fit,
                'predictions': sarimax_preds,
                'season': 7
            }
            print("Default SARIMAX model fitted successfully")
    except Exception as e:
        print(f"SARIMAX error: {e}")
    
    # 4. ETS (Exponential Smoothing) approach with optimized parameters
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Try different ETS configurations
        ets_configs = [
            {'trend': 'add', 'seasonal': 'add', 'damped': True},
            {'trend': 'add', 'seasonal': 'mul', 'damped': True},
            {'trend': 'mul', 'seasonal': 'add', 'damped': True},
            {'trend': 'mul', 'seasonal': 'mul', 'damped': False}
        ]
        
        best_ets_aic = float('inf')
        best_ets_model = None
        best_ets_preds = None
        best_ets_config = None
        
        for config in ets_configs:
            try:
                # Only use if we have enough data points
                if len(data) >= 14:
                    ets_model = ExponentialSmoothing(
                        data['Expenses'], 
                        seasonal_periods=7,  # Weekly seasonality
                        **config
                    )
                    ets_fit = ets_model.fit(optimized=True)
                    ets_preds = ets_fit.predict(start=0, end=len(data)-1)
                    
                    # Store if best model
                    if ets_fit.aic < best_ets_aic:
                        best_ets_aic = ets_fit.aic
                        best_ets_model = ets_fit
                        best_ets_preds = ets_preds
                        best_ets_config = config
            except Exception as e:
                print(f"ETS config {config} failed: {e}")
                continue
        
        # Store best ETS model
        if best_ets_model:
            results['ets'] = {
                'model': best_ets_model,
                'predictions': best_ets_preds,
                'config': best_ets_config
            }
            print(f"Best ETS model fitted successfully with config {best_ets_config}")
        else:
            # Fallback to default ETS
            ets_model = ExponentialSmoothing(
                data['Expenses'], 
                seasonal_periods=7,
                trend='add',
                seasonal='add',
                damped=True
            )
            ets_fit = ets_model.fit()
            ets_preds = ets_fit.predict(start=0, end=len(data)-1)
            
            results['ets'] = {
                'model': ets_fit,
                'predictions': ets_preds,
                'config': {'trend': 'add', 'seasonal': 'add', 'damped': True}
            }
            print("Default ETS model fitted successfully")
    except Exception as e:
        print(f"ETS error: {e}")
    
    # 5. Prophet Model (if available)
    if PROPHET_AVAILABLE and len(data) >= 14:
        try:
            # Prophet requires a specific data format
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data['Expenses']
            })
            
            # Configure Prophet with optimized parameters for expense data
            prophet_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,  # Only enable yearly seasonality with 1+ years of data
                seasonality_mode='additive',  # Changed to additive for more stable forecasts
                changepoint_prior_scale=0.05,  # Reduced to avoid overfitting
                interval_width=0.95,
                changepoint_range=0.8  # Look at more of the historical data
            )
            
            # Add custom seasonality for the start/middle/end of month patterns
            prophet_model.add_seasonality(
                name='monthly', 
                period=30.5, 
                fourier_order=3
            )
            
            # Add additional custom seasonalities if we have enough data
            if len(data) >= 30:
                # Try to capture bi-weekly patterns (common for paydays)
                prophet_model.add_seasonality(
                    name='biweekly', 
                    period=14, 
                    fourier_order=3
                )
            
            # Add country holidays if available for better handling of special days
            try:
                from holidays import country_holidays
                prophet_model.add_country_holidays(country_name='US')
                print("Added US holidays to Prophet model")
            except:
                pass
                
            # Add any additional available regressors from our features
            if len(data_with_features.columns) > 1:
                for col in data_with_features.columns:
                    if col not in ['Expenses', 'Category'] and col.startswith('is_') or col.startswith('day_'):
                        try:
                            prophet_data[col] = data_with_features[col].values
                            prophet_model.add_regressor(col)
                            print(f"Added regressor {col} to Prophet model")
                        except Exception as e:
                            print(f"Could not add regressor {col}: {e}")
            
            # Fit the model
            prophet_model.fit(prophet_data)
            
            # Make in-sample predictions
            prophet_forecast = prophet_model.predict(prophet_data)
            
            results['prophet'] = {
                'model': prophet_model,
                'predictions': prophet_forecast['yhat'].values,
                'data': prophet_data
            }
            print("Prophet model fitted successfully")
        except Exception as e:
            print(f"Prophet error: {e}")

    # 6. Add Moving Average Models (simple but effective for some patterns)
    try:
        # Simple Moving Average (SMA) model - often surprisingly effective
        window_sizes = [3, 7, 14, 30]
        best_ma_mae = float('inf')
        best_ma_window = None
        best_ma_preds = None
        
        for window in window_sizes:
            if len(data) > window:
                # Calculate MA predictions
                ma_series = data['Expenses'].rolling(window=window, min_periods=1).mean()
                ma_preds = ma_series.shift(1).fillna(data['Expenses'].mean())
                
                # Calculate error metrics
                mae = mean_absolute_error(data['Expenses'].values[window:], ma_preds.values[window:])
                
                if mae < best_ma_mae:
                    best_ma_mae = mae
                    best_ma_window = window
                    best_ma_preds = ma_preds
        
        if best_ma_preds is not None:
            results['moving_avg'] = {
                'predictions': best_ma_preds,
                'window': best_ma_window,
                'type': 'simple_ma'
            }
            print(f"Moving average model created with window {best_ma_window}")
        
        # Exponential Moving Average (EMA) - gives more weight to recent observations
        best_ema_mae = float('inf')
        best_ema_span = None
        best_ema_preds = None
        
        for span in [3, 7, 14, 30]:
            if len(data) > span:
                # Calculate EMA predictions
                ema_series = data['Expenses'].ewm(span=span, adjust=False).mean()
                ema_preds = ema_series.shift(1).fillna(data['Expenses'].mean())
                
                # Calculate error metrics
                mae = mean_absolute_error(data['Expenses'].values[span:], ema_preds.values[span:])
                
                if mae < best_ema_mae:
                    best_ema_mae = mae
                    best_ema_span = span
                    best_ema_preds = ema_preds
        
        if best_ema_preds is not None:
            results['exp_moving_avg'] = {
                'predictions': best_ema_preds,
                'span': best_ema_span,
                'type': 'exp_ma'
            }
            print(f"Exponential moving average model created with span {best_ema_span}")
    except Exception as e:
        print(f"Moving Average error: {e}")
    
    # 7. Try a basic Random Forest regressor for tabular data approach
    if len(data_with_features.columns) > 1:
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            # Prepare features and target
            X = data_with_features.drop('Expenses', axis=1)
            if 'Category' in X.columns:
                X = X.drop('Category', axis=1)
            
            # Handle categorical columns
            X_numeric = pd.get_dummies(X)
            y = data['Expenses'].values
            
            # Create lagged features (important for time series with ML models)
            for lag in [1, 3, 7]:
                if len(y) > lag:
                    lagged_values = np.roll(y, lag)
                    lagged_values[:lag] = np.mean(y)  # Fill initial values
                    X_numeric[f'lag_{lag}'] = lagged_values
            
            # Fit Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
            rf_model.fit(X_numeric, y)
            
            # Make predictions
            rf_preds = rf_model.predict(X_numeric)
            
            results['random_forest'] = {
                'model': rf_model,
                'predictions': rf_preds,
                'features': X_numeric.columns.tolist()
            }
            print("Random Forest model fitted successfully")
        except Exception as e:
            print(f"Random Forest error: {e}")
    
    # 8. Enhanced ensemble approach - weighted average based on model performance
    if len(results) > 1:
        try:
            # Evaluate all models first
            for model_name, model_info in results.items():
                actual = data['Expenses'].values
                predicted = model_info['predictions']
                
                # Make sure predictions align with actual values
                min_len = min(len(actual), len(predicted))
                actual = actual[:min_len]
                predicted = predicted[:min_len]
                
                # Calculate error metrics
                mae = mean_absolute_error(actual, predicted)
                
                # Calculate MAPE safely
                with np.errstate(divide='ignore', invalid='ignore'):
                    individual_mape = np.abs((actual - predicted) / actual) * 100
                
                # Handle infinity and NaN values
                individual_mape = individual_mape[~np.isinf(individual_mape) & ~np.isnan(individual_mape)]
                mape = np.mean(individual_mape) if len(individual_mape) > 0 else 100
                
                # Calculate accuracy
                accuracy = max(0, 100 - min(mape, 100))
                
                # Store metrics
                model_info['mae'] = mae
                model_info['mape'] = mape
                model_info['accuracy'] = accuracy
                
            # Get all predictions
            all_preds = []
            all_weights = []
            model_names = []
            
            # Calculate weights based on inverse of error metrics
            total_weight = 0
            for model_name, model_info in results.items():
                if 'accuracy' not in model_info:
                    continue
                    
                preds = model_info['predictions']
                if len(preds) == len(data):
                    # Use squared accuracy as weight to emphasize better models
                    weight = model_info['accuracy']**2
                    
                    # Boost weight for models with typically better forecasting properties
                    if model_name in ['prophet', 'auto_arima', 'sarimax']:
                        weight *= 1.2
                    
                    all_preds.append(preds)
                    all_weights.append(weight)
                    model_names.append(model_name)
                    total_weight += weight
            
            if all_preds and total_weight > 0:
                # Normalize weights
                all_weights = [w / total_weight for w in all_weights]
                
                # Apply weights and compute ensemble predictions
                weighted_preds = np.zeros(len(data))
                for preds, weight in zip(all_preds, all_weights):
                    weighted_preds += preds * weight
                
                results['ensemble'] = {
                    'predictions': weighted_preds,
                    'is_ensemble': True,
                    'models': model_names,
                    'weights': all_weights
                }
                print("Enhanced weighted ensemble model created successfully")
                print(f"Model weights: {dict(zip(model_names, all_weights))}")
        except Exception as e:
            print(f"Ensemble error: {e}")
    
    # Evaluate all models and select the best one
    for model_name, model_info in results.items():
        actual = data['Expenses'].values
        predicted = model_info['predictions']
        
        # Make sure predictions align with actual values
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # Calculate error metrics
        mae = mean_absolute_error(actual, predicted)
        
        # Calculate MAPE safely
        with np.errstate(divide='ignore', invalid='ignore'):
            individual_mape = np.abs((actual - predicted) / actual) * 100
        
        # Handle infinity and NaN values
        individual_mape = individual_mape[~np.isinf(individual_mape) & ~np.isnan(individual_mape)]
        mape = np.mean(individual_mape) if len(individual_mape) > 0 else 100
        
        # Calculate accuracy
        accuracy = max(0, 100 - min(mape, 100))
        
        # Store metrics
        model_info['mae'] = mae
        model_info['mape'] = mape
        model_info['accuracy'] = accuracy
        
        print(f"Model {model_name}: MAE={mae:.2f}, MAPE={mape:.2f}%, Accuracy={accuracy:.2f}%")
        
        # Track the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
    
    # If we have a valid model, use it for forecasting
    if best_model:
        print(f"Selected best model: {best_model} with accuracy: {best_accuracy:.2f}%")
        
        # Forecast next 30 days
        forecast_steps = 30
        current_date = datetime.datetime.now().date()
        next_day = current_date + pd.DateOffset(days=1)
        forecast_index = pd.date_range(start=next_day, periods=forecast_steps, freq='D')
        
        # Generate forecast using the selected model
        if best_model == 'prophet':
            # Prophet requires special handling
            future = pd.DataFrame({'ds': forecast_index})
            prophet_forecast = results['prophet']['model'].predict(future)
            forecast = prophet_forecast['yhat'].values
            
            # Create a visualization with Prophet components
            fig = make_subplots(rows=3, cols=1, 
                              subplot_titles=["Expense Forecast", "Weekly Pattern", "Trend Analysis"],
                              vertical_spacing=0.1,
                              specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]])
            
            # Plot 1: Actual vs Forecasted
            fig.add_trace(go.Scatter(x=data.index, y=data['Expenses'], 
                                   mode='lines+markers', name='Previous Expenses'), row=1, col=1)
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast, 
                                   mode='lines+markers', name='Forecasted Expenses',
                                   line=dict(color='firebrick', dash='dot')), row=1, col=1)
            
            # Add uncertainty intervals
            fig.add_trace(go.Scatter(
                x=forecast_index, 
                y=prophet_forecast['yhat_upper'], 
                fill=None, 
                mode='lines', 
                line=dict(color='rgba(200, 0, 0, 0.2)'), 
                showlegend=False), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=forecast_index, 
                y=prophet_forecast['yhat_lower'], 
                fill='tonexty', 
                mode='lines', 
                line=dict(color='rgba(200, 0, 0, 0.2)'), 
                name='95% Confidence Interval'), row=1, col=1)
            
            # If we have Prophet components, show them
            model_data = results['prophet']['data']
            model = results['prophet']['model']
            
            # Plot 2: Weekly Pattern
            prophet_forecast = model.predict(model_data)
            if 'weekly' in prophet_forecast.columns:
                weekly_component = prophet_forecast['weekly']
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                day_indices = prophet_forecast['ds'].dt.dayofweek
                
                # Group by day of week and calculate average
                weekly_avg = []
                for i in range(7):
                    day_data = weekly_component[day_indices == i]
                    if not day_data.empty:
                        weekly_avg.append(day_data.mean())
                    else:
                        weekly_avg.append(0)
                        
                fig.add_trace(go.Bar(x=days, y=weekly_avg, name='Weekly Pattern'), row=2, col=1)
            
            # Plot 3: Trend Analysis
            if 'trend' in prophet_forecast.columns:
                fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['trend'], 
                                       mode='lines', name='Trend Component'), row=3, col=1)
            
            # Update layout
            fig.update_layout(height=800, title_text="Expense Forecast with Pattern Analysis")
            
        elif best_model == 'ets':
            # Exponential Smoothing model
            ets_model = results['ets']['model']
            forecast = ets_model.forecast(steps=forecast_steps)
            
            # Ensure no negative forecasts
            forecast = np.maximum(forecast, 0)
            
            # Create a standard visualization with decomposition
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=["Expense Forecast", "Seasonal Pattern"],
                              vertical_spacing=0.1)
            
            # Plot 1: Actual vs Forecasted
            fig.add_trace(go.Scatter(x=data.index, y=data['Expenses'], 
                                   mode='lines+markers', name='Previous Expenses'), row=1, col=1)
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast, 
                                   mode='lines+markers', name='Forecasted Expenses', 
                                   line=dict(color='firebrick', dash='dot')), row=1, col=1)
            
            # Plot 2: Seasonal decomposition
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomposition = seasonal_decompose(data['Expenses'], model='additive', period=7)
                fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, 
                                       mode='lines', name='Seasonal Component'), row=2, col=1)
            except Exception as e:
                print(f"Error in seasonal decomposition: {e}")
                
        elif best_model == 'ensemble':
            # Generate forecast for each component model and combine with weights
            model_forecasts = {}
            weights = results['ensemble']['weights']
            model_names = results['ensemble']['models']
            
            forecast = np.zeros(forecast_steps)
            for i, model_name in enumerate(model_names):
                # Skip the ensemble model itself
                if model_name == 'ensemble':
                    continue
                    
                model_weight = weights[i]
                
                try:
                    if model_name == 'prophet':
                        future = pd.DataFrame({'ds': forecast_index})
                        prophet_forecast = results[model_name]['model'].predict(future)
                        model_forecast = prophet_forecast['yhat'].values
                    elif model_name == 'ets':
                        model_forecast = results[model_name]['model'].forecast(steps=forecast_steps)
                    elif model_name == 'auto_arima':
                        model_forecast = results[model_name]['model'].predict(n_periods=forecast_steps)
                    elif model_name in ['moving_avg', 'exp_moving_avg']:
                        # For moving average models, use the pattern from the most recent period
                        recent_pattern = data['Expenses'][-forecast_steps:].values
                        if len(recent_pattern) < forecast_steps:
                            # Pad with the mean if not enough data
                            pad_size = forecast_steps - len(recent_pattern)
                            recent_pattern = np.pad(recent_pattern, (0, pad_size), 
                                                   'constant', constant_values=data['Expenses'].mean())
                        model_forecast = recent_pattern
                    elif model_name == 'random_forest':
                        # For Random Forest, need to create future feature values
                        rf_model = results[model_name]['model']
                        features = results[model_name]['features']
                        
                        # Create future feature data
                        future_features = pd.DataFrame(index=forecast_index)
                        for feature in features:
                            if feature.startswith('lag_'):
                                lag = int(feature.split('_')[1])
                                if lag == 1:
                                    # For lag 1, use the most recent actual value
                                    future_features[feature] = np.pad(data['Expenses'].values[-lag:], 
                                                                     (0, forecast_steps-lag), 'edge')
                                else:
                                    # For other lags, use historical patterns
                                    future_features[feature] = np.pad(data['Expenses'].values[-lag:], 
                                                                     (0, forecast_steps-lag), 'mean')
                            elif feature.startswith('day_of_week'):
                                future_features[feature] = forecast_index.dayofweek
                            elif feature.startswith('day_of_month'):
                                future_features[feature] = forecast_index.day
                            elif feature.startswith('month'):
                                future_features[feature] = forecast_index.month
                            elif feature.startswith('is_weekend'):
                                future_features[feature] = (forecast_index.dayofweek >= 5).astype(int)
                            elif feature.startswith('is_holiday'):
                                # Default to not a holiday if we can't determine
                                future_features[feature] = 0
                            else:
                                # Fill with zeros for other features
                                future_features[feature] = 0
                        
                        # Make the prediction
                        model_forecast = rf_model.predict(future_features)
                    else:
                        # ARIMA, SARIMAX
                        model_forecast = results[model_name]['model'].forecast(steps=forecast_steps)
                        
                    # Store the forecast and add to ensemble with weight
                    model_forecasts[model_name] = model_forecast
                    forecast += model_forecast * model_weight
                except Exception as e:
                    print(f"Error forecasting with {model_name}: {e}")
            
            # Ensure no negative forecasts and apply reasonable bounds
            forecast = np.maximum(forecast, 0)
            max_historical = data['Expenses'].max()
            forecast = np.minimum(forecast, max_historical * 2.0)
            
            # Create visualization
            fig = go.Figure()
            
            # Add traces for historical and forecasted expenses
            fig.add_trace(go.Scatter(x=data.index, y=data['Expenses'], 
                                   mode='lines+markers', name='Previous Expenses',
                                   line=dict(color='royalblue')))
                                   
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast, 
                                   mode='lines+markers', name='Forecasted Expenses (Ensemble)', 
                                   line=dict(color='firebrick', width=3)))
            
            # Add individual model forecasts with lower opacity
            for model_name, model_forecast in model_forecasts.items():
                if len(model_forecast) == len(forecast_index):
                    fig.add_trace(go.Scatter(
                        x=forecast_index, 
                        y=model_forecast,
                        mode='lines', 
                        name=f'{model_name} Forecast',
                        line=dict(width=1, dash='dot'),
                        opacity=0.5
                    ))
            
            # Create confidence bands based on model variance
            if len(model_forecasts) > 1:
                all_forecasts = np.array([f for f in model_forecasts.values() if len(f) == len(forecast_index)])
                if len(all_forecasts) > 1:
                    forecast_std = np.std(all_forecasts, axis=0)
                    upper_bound = forecast + 1.96 * forecast_std
                    lower_bound = np.maximum(forecast - 1.96 * forecast_std, 0)
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_index,
                        y=upper_bound,
                        fill=None,
                        mode='lines',
                        line=dict(color='rgba(200, 0, 0, 0.0)'),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_index,
                        y=lower_bound,
                        fill='tonexty',
                        mode='lines',
                        line=dict(color='rgba(200, 0, 0, 0.0)'),
                        name='95% Confidence Interval'
                    ))
        else:
            # For standard models (ARIMA, SARIMAX, etc.)
            try:
                model = results[best_model]['model']
                if best_model == 'auto_arima':
                    forecast = model.predict(n_periods=forecast_steps)
                elif best_model in ['moving_avg', 'exp_moving_avg']:
                    # Use recent patterns for MA models
                    if len(data) >= forecast_steps:
                        # Use the most recent period as forecast
                        forecast = data['Expenses'][-forecast_steps:].values
                    else:
                        # If not enough data, repeat the pattern
                        forecast = np.tile(data['Expenses'].values, forecast_steps // len(data) + 1)[:forecast_steps]
                else:
                    forecast = model.forecast(steps=forecast_steps)
            except Exception as e:
                print(f"Error forecasting with {best_model}: {e}")
                # Fallback to a simple AR(1) forecast
                arima_simple = ARIMA(data['Expenses'], order=(1, 0, 0))
                forecast = arima_simple.fit().forecast(steps=forecast_steps)
            
            # Ensure no negative forecasts and realistic values
            forecast = np.maximum(forecast, 0)
            max_historical = data['Expenses'].max()
            forecast = np.minimum(forecast, max_historical * 2.0)
            
            # Create a standard visualization
            fig = go.Figure()
            
            # Add traces for historical and forecasted expenses
            fig.add_trace(go.Scatter(x=data.index, y=data['Expenses'], 
                                   mode='lines+markers', name='Previous Expenses'))
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast, 
                                   mode='lines+markers', name='Forecasted Expenses', 
                                   line=dict(color='firebrick', dash='dot')))
        
        # Update layout with better styling
        fig.update_layout(
            title='Expense Forecast for Next 30 Days',
            xaxis_title='Date',
            yaxis_title='Expenses (₹)',
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Create a div with the plot
        plot_div = plot(fig, output_type='div', include_plotlyjs=True)
        
        # Create DataFrame for forecasted expenses
        forecast_data = pd.DataFrame({'Date': forecast_index, 'Forecasted_Expenses': forecast})
        forecast_data_list = forecast_data.to_dict(orient='records')
        
        # Calculate total forecasted
        total_forecasted_expenses = np.sum(forecast)
        
        # Get metrics from best model
        mae = results[best_model]['mae']
        mape = results[best_model]['mape']
        accuracy = results[best_model]['accuracy']
        
        # Calculate better category forecasts if we have categorial data
        if 'Category' in data.columns:
            category_forecasts_dict = forecast_categories_advanced(data, forecast_index, forecast)
        else:
            category_forecasts_dict = category_forecasts(data)
        
        # Print metrics
        print(f"Model Evaluation for {username}'s Expense Forecast:")
        print(f"Best Model: {best_model}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Forecast Accuracy: {accuracy:.2f}%")
        print("-" * 50)
        
        return {
            'forecast_data': forecast_data_list,
            'total_forecasted_expenses': round(total_forecasted_expenses, 2),
            'category_forecasts': category_forecasts_dict,
            'model_accuracy': round(accuracy, 2),
            'mean_absolute_error': round(mae, 2),
            'mape': round(mape, 2),
            'plot_div': plot_div,
            'model_name': best_model
        }
    
    # Fallback to simple forecast if no model worked
    return simple_forecast(data, username)
