# finances/tasks.py
from celery import shared_task
from expenses.models import Expense
from userincome.models import UserIncome
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.cache import cache
from django.db.models import Sum, Avg
import pandas as pd
import numpy as np
from datetime import timedelta

@shared_task
def generate_expense_report(user_id, report_type='monthly', date_range=30, use_cache=True):
    """
    Generate expense report asynchronously with optimizations for handling large datasets
    """
    # Check cache first if enabled
    cache_key = f"expense_report_{user_id}_{report_type}_{date_range}"
    if use_cache:
        cached_report = cache.get(cache_key)
        if cached_report:
            return cached_report

    try:
        user = User.objects.get(pk=user_id)
        
        # Determine date range
        end_date = timezone.now().date()
        if report_type == 'monthly':
            start_date = end_date - timedelta(days=30)
        elif report_type == 'quarterly':
            start_date = end_date - timedelta(days=90)
        elif report_type == 'yearly':
            start_date = end_date - timedelta(days=365)
        else:
            # Custom date range
            start_date = end_date - timedelta(days=date_range)
            
        # Get expenses within date range
        # For very large datasets, we'll use database aggregations instead of loading all records
        expense_data = {}
        
        # Get expense totals by category - database aggregation
        expense_by_category = Expense.objects.filter(
            owner=user, 
            date__gte=start_date,
            date__lte=end_date
        ).values('category').annotate(
            total=Sum('amount')
        ).order_by('-total')
        
        expense_data['by_category'] = list(expense_by_category)
        
        # Get expense totals by date - database aggregation
        expense_by_date = Expense.objects.filter(
            owner=user,
            date__gte=start_date,
            date__lte=end_date
        ).values('date').annotate(
            total=Sum('amount')
        ).order_by('date')
        
        # Convert to list and fill in missing dates for continuous data
        date_data = list(expense_by_date)
        
        # If we have a lot of dates, sample them to reduce dataset size
        # For monthly reports, daily granularity is fine
        # For quarterly/yearly reports, we can sample to weekly data points
        if report_type == 'quarterly' and len(date_data) > 30:
            # Convert to pandas for efficient resampling
            df = pd.DataFrame(date_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            # Resample to weekly data points
            weekly_data = df.resample('W').sum().reset_index()
            date_data = weekly_data.to_dict('records')
        elif report_type == 'yearly' and len(date_data) > 52:
            # Convert to pandas for efficient resampling
            df = pd.DataFrame(date_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            # Resample to bi-weekly data points for yearly reports
            biweekly_data = df.resample('2W').sum().reset_index()
            date_data = biweekly_data.to_dict('records')
            
        expense_data['by_date'] = date_data
        
        # Get expense totals by payment method - database aggregation
        expense_by_payment = Expense.objects.filter(
            owner=user,
            date__gte=start_date,
            date__lte=end_date
        ).values('payment_method').annotate(
            total=Sum('amount')
        ).order_by('-total')
        
        expense_data['by_payment'] = list(expense_by_payment)
        
        # Calculate summary statistics
        summary = {
            'total': Expense.objects.filter(
                owner=user,
                date__gte=start_date,
                date__lte=end_date
            ).aggregate(Sum('amount'))['amount__sum'] or 0,
            
            'avg_daily': Expense.objects.filter(
                owner=user,
                date__gte=start_date,
                date__lte=end_date
            ).values('date').annotate(
                daily=Sum('amount')
            ).aggregate(Avg('daily'))['daily__avg'] or 0,
            
            'count': Expense.objects.filter(
                owner=user,
                date__gte=start_date,
                date__lte=end_date
            ).count()
        }
        
        expense_data['summary'] = summary
        
        # Calculate expense trend - if data is large, use sampling
        if expense_data['by_date']:
            if len(expense_data['by_date']) > 60:  # If we have a lot of data points
                # Sample to ensure manageable data size
                trend_data = expense_data['by_date']
                sample_size = min(60, len(trend_data))
                indices = np.linspace(0, len(trend_data)-1, sample_size, dtype=int)
                trend_data = [trend_data[i] for i in indices]
                expense_data['trend_data'] = trend_data
            else:
                expense_data['trend_data'] = expense_data['by_date']
        
        # Cache the results if caching is enabled
        if use_cache:
            # Cache duration depends on report type
            if report_type == 'monthly':
                cache_time = 60 * 60 * 6  # 6 hours for monthly reports
            elif report_type == 'quarterly':
                cache_time = 60 * 60 * 12  # 12 hours for quarterly
            else:
                cache_time = 60 * 60 * 24  # 24 hours for yearly
                
            cache.set(cache_key, expense_data, cache_time)
        
        return expense_data
        
    except User.DoesNotExist:
        return {'error': 'User not found'}
    except Exception as e:
        return {'error': str(e)}

@shared_task
def generate_income_report(user_id, report_type='monthly', date_range=30, use_cache=True):
    """
    Generate income report asynchronously with optimizations for handling large datasets
    """
    # Check cache first if enabled
    cache_key = f"income_report_{user_id}_{report_type}_{date_range}"
    if use_cache:
        cached_report = cache.get(cache_key)
        if cached_report:
            return cached_report
    
    try:
        user = User.objects.get(pk=user_id)
        
        # Determine date range
        end_date = timezone.now().date()
        if report_type == 'monthly':
            start_date = end_date - timedelta(days=30)
        elif report_type == 'quarterly':
            start_date = end_date - timedelta(days=90)
        elif report_type == 'yearly':
            start_date = end_date - timedelta(days=365)
        else:
            # Custom date range
            start_date = end_date - timedelta(days=date_range)
            
        # Get income within date range - use DB aggregation for efficiency
        income_data = {}
        
        # Income by source
        income_by_source = UserIncome.objects.filter(
            owner=user,
            date__gte=start_date,
            date__lte=end_date
        ).values('source').annotate(
            total=Sum('amount')
        ).order_by('-total')
        
        income_data['by_source'] = list(income_by_source)
        
        # Income by date
        income_by_date = UserIncome.objects.filter(
            owner=user,
            date__gte=start_date,
            date__lte=end_date
        ).values('date').annotate(
            total=Sum('amount')
        ).order_by('date')
        
        # Convert to list and optimize for large datasets
        date_data = list(income_by_date)
        
        # If we have a lot of dates, sample them to reduce dataset size
        if report_type == 'quarterly' and len(date_data) > 30:
            # Convert to pandas for efficient resampling
            df = pd.DataFrame(date_data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                # Resample to weekly data points
                weekly_data = df.resample('W').sum().reset_index()
                date_data = weekly_data.to_dict('records')
        elif report_type == 'yearly' and len(date_data) > 52:
            # Convert to pandas for efficient resampling
            df = pd.DataFrame(date_data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                # Resample to bi-weekly data points for yearly reports
                biweekly_data = df.resample('2W').sum().reset_index()
                date_data = biweekly_data.to_dict('records')
            
        income_data['by_date'] = date_data
        
        # Summary statistics
        summary = {
            'total': UserIncome.objects.filter(
                owner=user,
                date__gte=start_date,
                date__lte=end_date
            ).aggregate(Sum('amount'))['amount__sum'] or 0,
            
            'avg_monthly': (UserIncome.objects.filter(
                owner=user,
                date__gte=start_date,
                date__lte=end_date
            ).aggregate(Sum('amount'))['amount__sum'] or 0) / (date_range / 30),
            
            'count': UserIncome.objects.filter(
                owner=user,
                date__gte=start_date,
                date__lte=end_date
            ).count()
        }
        
        income_data['summary'] = summary
        
        # Calculate income trend - if data is large, use sampling
        if income_data['by_date']:
            if len(income_data['by_date']) > 60:  # If we have a lot of data points
                # Sample to ensure manageable data size
                trend_data = income_data['by_date']
                sample_size = min(60, len(trend_data))
                indices = np.linspace(0, len(trend_data)-1, sample_size, dtype=int)
                trend_data = [trend_data[i] for i in indices]
                income_data['trend_data'] = trend_data
            else:
                income_data['trend_data'] = income_data['by_date']
        
        # Cache the results if caching is enabled
        if use_cache:
            # Cache duration depends on report type
            if report_type == 'monthly':
                cache_time = 60 * 60 * 6  # 6 hours for monthly reports
            elif report_type == 'quarterly':
                cache_time = 60 * 60 * 12  # 12 hours for quarterly
            else:
                cache_time = 60 * 60 * 24  # 24 hours for yearly
                
            cache.set(cache_key, income_data, cache_time)
        
        return income_data
        
    except User.DoesNotExist:
        return {'error': 'User not found'}
    except Exception as e:
        return {'error': str(e)}

@shared_task
def generate_combined_financial_report(user_id, report_type='monthly', date_range=30):
    """
    Generate a combined report with both income and expense data
    """
    # Get expense and income reports (which are already optimized)
    expense_data = generate_expense_report(user_id, report_type, date_range)
    income_data = generate_income_report(user_id, report_type, date_range)
    
    # Combine the reports
    combined_report = {
        'expenses': expense_data,
        'income': income_data,
        'summary': {
            'net': (income_data.get('summary', {}).get('total', 0) - 
                    expense_data.get('summary', {}).get('total', 0)),
            'savings_rate': calculate_savings_rate(income_data, expense_data),
        }
    }
    
    return combined_report

def calculate_savings_rate(income_data, expense_data):
    """Calculate the savings rate as a percentage"""
    total_income = income_data.get('summary', {}).get('total', 0)
    total_expenses = expense_data.get('summary', {}).get('total', 0)
    
    if total_income == 0:
        return 0
    
    savings = total_income - total_expenses
    savings_rate = (savings / total_income) * 100
    
    return round(savings_rate, 2)
