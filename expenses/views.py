from django.shortcuts import render, redirect, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from datetime import datetime, date, timedelta
from .models import Category, Expense, ExpenseLimit
from django.contrib.auth.models import User
from django.core.paginator import Paginator
import json
from django.http import JsonResponse
from userpreferences.models import UserPreference
from userprofile.models import Profile
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from django.core.mail import send_mail
from django.conf import settings
from django.db.models import Sum, Avg

data = pd.read_csv('dataset.csv')

# Preprocessing
try:
    nltk.download('stopwords')  # Ensure stopwords corpus is downloaded
    stop_words = set(stopwords.words('english'))
except Exception as e:
    stop_words = set()  # Fallback to an empty set if download fails
    print(f"Error downloading stopwords: {str(e)}")

def home(request):
    return render(request, 'expenses/landing.html')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

data['clean_description'] = data['description'].apply(preprocess_text)

# Feature extraction
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['clean_description'])

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, data['category'])

@login_required(login_url='/authentication/login')
def search_expenses(request):
    if request.method == 'POST':
        search_str = json.loads(request.body).get('searchText')
        expenses = Expense.objects.filter(
            amount__istartswith=search_str, owner=request.user) | Expense.objects.filter(
            date__istartswith=search_str, owner=request.user) | Expense.objects.filter(
            description__icontains=search_str, owner=request.user) | Expense.objects.filter(
            category__icontains=search_str, owner=request.user)
        
        data = []
        for expense in expenses:
            # Get the user object for the spent_by username
            try:
                spent_by_user = User.objects.get(username=expense.spent_by)
                spent_by_display = spent_by_user.get_full_name() or spent_by_user.username
                if spent_by_user == request.user:
                    spent_by_display += " (Self)"
                elif hasattr(spent_by_user, 'profile'):
                    spent_by_display += f" ({spent_by_user.profile.relationship})"
            except User.DoesNotExist:
                spent_by_display = expense.spent_by

            data.append({
                'amount': expense.amount,
                'category': expense.category,
                'description': expense.description,
                'date': expense.date,
                'id': expense.id,
                'spent_by': spent_by_display
            })
        
        return JsonResponse(data, safe=False)

@login_required(login_url='/authentication/login')
def index(request):
    expenses = Expense.get_user_viewable_expenses(request.user)
    
    # Enhance expenses with spent_by display names
    for expense in expenses:
        try:
            spent_by_user = User.objects.get(username=expense.spent_by)
            spent_by_display = spent_by_user.get_full_name() or spent_by_user.username
            if spent_by_user == request.user:
                spent_by_display += " (Self)"
            elif hasattr(spent_by_user, 'profile'):
                spent_by_display += f" ({spent_by_user.profile.relationship})"
            expense.spent_by_display = spent_by_display
        except User.DoesNotExist:
            expense.spent_by_display = expense.spent_by

    paginator = Paginator(expenses, 5)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    try:
        user_preferences = UserPreference.objects.get(user=request.user)
        currency = user_preferences.currency
    except UserPreference.DoesNotExist:
        currency = "INR - Indian Rupee"

    context = {
        'expenses': expenses,
        'page_obj': page_obj,
        'currency': currency,
    }
    return render(request, 'expenses/index.html', context)

daily_expense_amounts = {}

def send_limit_notification(user, amount):
    try:
        subject = 'Daily Expense Limit Exceeded'
        message = f'Your daily expenses ({amount}) have exceeded your set limit.'
        from_email = settings.EMAIL_HOST_USER
        recipient_list = [user.email]
        send_mail(subject, message, from_email, recipient_list)
        return True
    except Exception as e:
        print(f"Error sending notification: {str(e)}")
        return False

def get_expense_of_day(user):
    today = date.today()
    expenses = Expense.objects.filter(owner=user, date=today)
    return sum(expense.amount for expense in expenses)

@login_required(login_url='/authentication/login')
def add_expense(request):
    categories = Category.objects.all()
    spent_by_choices = Expense.get_spent_by_choices(request.user)
    payment_method_choices = Expense.PAYMENT_METHOD_CHOICES
    transaction_category_choices = Expense.TRANSACTION_CATEGORY_CHOICES
    
    context = {
        'categories': categories,
        'values': request.POST,
        'spent_by_choices': spent_by_choices,
        'payment_method_choices': payment_method_choices,
        'transaction_category_choices': transaction_category_choices
    }

    if request.method == 'GET':
        return render(request, 'expenses/add_expense.html', context)

    if request.method == 'POST':
        amount = request.POST['amount']
        description = request.POST['description']
        category = request.POST['category']
        date = request.POST.get('expense_date', timezone.now)
        spent_by = request.POST.get('spent_by', 'Self')
        payment_method = request.POST.get('payment_method', 'CASH')
        transaction_category = request.POST.get('transaction_category', 'OTHER')

        if not amount:
            messages.error(request, 'Amount is required')
            return render(request, 'expenses/add_expense.html', context)

        if not description:
            messages.error(request, 'Description is required')
            return render(request, 'expenses/add_expense.html', context)

        # Create expense with all fields
        expense = Expense.objects.create(
            amount=amount,
            date=date,
            category=category,
            description=description,
            owner=request.user,
            spent_by=spent_by,
            payment_method=payment_method,
            transaction_category=transaction_category
        )
        
        messages.success(request, 'Expense saved successfully')
        return redirect('expenses')

@login_required(login_url='/authentication/login')
def expense_edit(request, id):
    expense = Expense.objects.get(pk=id)
    categories = Category.objects.all()
    spent_by_choices = Expense.get_spent_by_choices(request.user)
    payment_method_choices = Expense.PAYMENT_METHOD_CHOICES
    transaction_category_choices = Expense.TRANSACTION_CATEGORY_CHOICES
    
    context = {
        'expense': expense,
        'values': expense,
        'categories': categories,
        'spent_by_choices': spent_by_choices,
        'payment_method_choices': payment_method_choices,
        'transaction_category_choices': transaction_category_choices
    }

    if request.method == 'GET':
        return render(request, 'expenses/edit-expense.html', context)

    if request.method == 'POST':
        amount = request.POST['amount']
        description = request.POST['description']
        category = request.POST['category']
        date_str = request.POST.get('expense_date')
        spent_by = request.POST.get('spent_by', 'Self')
        payment_method = request.POST.get('payment_method', expense.payment_method)
        transaction_category = request.POST.get('transaction_category', expense.transaction_category)

        if not amount:
            messages.error(request, 'Amount is required')
            return render(request, 'expenses/edit-expense.html', context)

        if not description:
            messages.error(request, 'Description is required')
            return render(request, 'expenses/edit-expense.html', context)

        try:
            expense_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            today = date.today()

            if expense_date > today:
                messages.error(request, 'Date cannot be in the future')
                return render(request, 'expenses/edit-expense.html', context)

            expense.owner = request.user
            expense.amount = amount
            expense.date = expense_date
            expense.category = category
            expense.description = description
            expense.spent_by = spent_by
            expense.payment_method = payment_method
            expense.transaction_category = transaction_category

            expense.save()
            messages.success(request, 'Expense updated successfully')
            return redirect('expenses')
        except ValueError as e:
            messages.error(request, f'Invalid date format: {str(e)}')
            return render(request, 'expenses/edit-expense.html', context)

@login_required(login_url='/authentication/login')
def delete_expense(request, id):
    expense = Expense.objects.get(pk=id)
    expense.delete()
    messages.success(request, 'Expense removed')
    return redirect('expenses')

@login_required(login_url='/authentication/login')
def expense_category_summary(request):
    todays_date = date.today()
    six_months_ago = todays_date - timedelta(days=30*6)
    expenses = Expense.objects.filter(owner=request.user,
                                      date__gte=six_months_ago, date__lte=todays_date)
    finalrep = {}
    spent_by_rep = {}
    payment_method_rep = {}
    transaction_category_rep = {}

    def get_category(expense):
        return expense.category
    
    def get_spent_by(expense):
        return expense.spent_by
        
    def get_payment_method(expense):
        return expense.payment_method
        
    def get_transaction_category(expense):
        return expense.transaction_category

    category_list = list(set(map(get_category, expenses)))
    spent_by_list = list(set(map(get_spent_by, expenses)))
    payment_method_list = list(set(map(get_payment_method, expenses)))
    transaction_category_list = list(set(map(get_transaction_category, expenses)))

    def get_expense_category_amount(category):
        amount = 0
        filtered_by_category = expenses.filter(category=category)
        for item in filtered_by_category:
            amount += item.amount
        return amount

    def get_expense_spent_by_amount(spent_by):
        amount = 0
        filtered_by_spent = expenses.filter(spent_by=spent_by)
        for item in filtered_by_spent:
            amount += item.amount
        return amount
        
    def get_expense_payment_method_amount(payment_method):
        amount = 0
        filtered_by_payment = expenses.filter(payment_method=payment_method)
        for item in filtered_by_payment:
            amount += item.amount
        return amount
        
    def get_expense_transaction_category_amount(transaction_category):
        amount = 0
        filtered_by_transaction = expenses.filter(transaction_category=transaction_category)
        for item in filtered_by_transaction:
            amount += item.amount
        return amount

    # Get display names for the payment methods and transaction categories
    payment_method_display = {code: label for code, label in Expense.PAYMENT_METHOD_CHOICES}
    transaction_category_display = {code: label for code, label in Expense.TRANSACTION_CATEGORY_CHOICES}
    
    for y in category_list:
        finalrep[y] = get_expense_category_amount(y)
    
    for y in spent_by_list:
        spent_by_rep[y] = get_expense_spent_by_amount(y)
        
    for y in payment_method_list:
        display_name = payment_method_display.get(y, y)
        payment_method_rep[display_name] = get_expense_payment_method_amount(y)
        
    for y in transaction_category_list:
        display_name = transaction_category_display.get(y, y)
        transaction_category_rep[display_name] = get_expense_transaction_category_amount(y)

    return JsonResponse({
        'expense_category_data': finalrep,
        'expense_spent_by_data': spent_by_rep,
        'payment_method_data': payment_method_rep,
        'transaction_category_data': transaction_category_rep
    }, safe=False)

@login_required(login_url='/authentication/login')
def stats_view(request):
    # Get date range from request, default to 30 days
    date_range = request.GET.get('date_range', '30')
    
    # Calculate date range
    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=int(date_range))
    
    # Get expenses within date range
    expenses = Expense.objects.filter(
        owner=request.user,
        date__gte=start_date,
        date__lte=end_date
    )
    
    # Calculate total expenses with fallback to 0
    total_expenses = expenses.aggregate(
        total=Sum('amount')
    )['total'] or 0.00
    
    # Calculate monthly average (handle zero case)
    try:
        monthly_average = total_expenses / (int(date_range) / 30)
    except (ZeroDivisionError, TypeError):
        monthly_average = 0.00
    
    # Get demographic data if available
    try:
        user_profile = request.user.profile
        demographics = {
            'gender': user_profile.get_gender_display(),
            'age': calculate_age(user_profile.date_of_birth) if user_profile.date_of_birth else None,
        }
    except:
        demographics = {'gender': None, 'age': None}
    
    # Get top categories (handle empty case)
    top_categories = expenses.values('category')\
        .annotate(total=Sum('amount'))\
        .order_by('-total')[:5]
    
    # Get payment method analysis
    payment_methods = expenses.values('payment_method')\
        .annotate(total=Sum('amount'))\
        .order_by('-total')
    
    # Get transaction category analysis
    transaction_categories = expenses.values('transaction_category')\
        .annotate(total=Sum('amount'))\
        .order_by('-total')
    
    # Convert code values to display values
    payment_method_display = {code: label for code, label in Expense.PAYMENT_METHOD_CHOICES}
    transaction_category_display = {code: label for code, label in Expense.TRANSACTION_CATEGORY_CHOICES}
    
    formatted_payment_methods = []
    for pm in payment_methods:
        formatted_payment_methods.append({
            'name': payment_method_display.get(pm['payment_method'], pm['payment_method']),
            'total': pm['total'] or 0
        })
    
    formatted_transaction_categories = []
    for tc in transaction_categories:
        formatted_transaction_categories.append({
            'name': transaction_category_display.get(tc['transaction_category'], tc['transaction_category']),
            'total': tc['total'] or 0
        })
    
    # Prepare chart data for categories
    categories = []
    amounts = []
    
    for category in top_categories:
        categories.append(category['category'])
        amounts.append(float(category['total'] or 0))
    
    chart_data = {
        'labels': categories,
        'datasets': [{
            'label': 'Expenses by Category',
            'data': amounts,
            'backgroundColor': [
                '#FF6384', '#36A2EB', '#FFCE56',
                '#4BC0C0', '#9966FF'
            ]
        }]
    }
    
    # Prepare chart data for payment methods
    payment_labels = []
    payment_amounts = []
    
    for pm in formatted_payment_methods[:5]:  # Top 5 payment methods
        payment_labels.append(pm['name'])
        payment_amounts.append(float(pm['total']))
    
    payment_chart_data = {
        'labels': payment_labels,
        'datasets': [{
            'label': 'Expenses by Payment Method',
            'data': payment_amounts,
            'backgroundColor': [
                '#8b5cf6', '#ec4899', '#3b82f6', 
                '#10b981', '#f59e0b'
            ]
        }]
    }
    
    # Prepare chart data for transaction categories
    transaction_labels = []
    transaction_amounts = []
    
    for tc in formatted_transaction_categories[:5]:  # Top 5 transaction categories
        transaction_labels.append(tc['name'])
        transaction_amounts.append(float(tc['total']))
    
    transaction_chart_data = {
        'labels': transaction_labels,
        'datasets': [{
            'label': 'Expenses by Transaction Category',
            'data': transaction_amounts,
            'backgroundColor': [
                '#6366f1', '#14b8a6', '#f97316',
                '#8b5cf6', '#06b6d4'
            ]
        }]
    }
    
    context = {
        'total_expenses': total_expenses,
        'monthly_average': round(monthly_average, 2),
        'top_categories': [
            {'name': cat['category'], 'total': cat['total'] or 0} 
            for cat in top_categories
        ],
        'payment_methods': formatted_payment_methods,
        'transaction_categories': formatted_transaction_categories,
        'demographics': demographics,
        'date_range': date_range,
        'chart_data': json.dumps(chart_data),
        'payment_chart_data': json.dumps(payment_chart_data),
        'transaction_chart_data': json.dumps(transaction_chart_data)
    }
    
    return render(request, 'expenses/stats.html', context)

def calculate_age(birth_date):
    """Calculate age from birth date"""
    if not birth_date:
        return None
    
    today = date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

@login_required(login_url='/authentication/login')
def predict_category(description):
    predict_category_url = 'http://localhost:8000/api/predict-category/'  # Use the correct URL path
    data = {'description': description}
    response = requests.post(predict_category_url, data=data)

    if response.status_code == 200:
        # Get the predicted category from the response
        predicted_category = response.json().get('predicted_category')
        return predicted_category
    else:
        # Handle the case where the prediction request failed
        return None
    

def set_expense_limit(request):
    if request.method == "POST":
        daily_expense_limit = request.POST.get('daily_expense_limit')
        
        existing_limit = ExpenseLimit.objects.filter(owner=request.user).first()
        
        if existing_limit:
            existing_limit.daily_expense_limit = daily_expense_limit
            existing_limit.save()
        else:
            ExpenseLimit.objects.create(owner=request.user, daily_expense_limit=daily_expense_limit)
        
        messages.success(request, "Daily Expense Limit Updated Successfully!")
        return HttpResponseRedirect('/preferences/')
    else:
        return HttpResponseRedirect('/preferences/')

def about(request):
    return render(request, 'expenses/about.html')

def contact(request):
    return render(request, 'expenses/contact.html')

def privacy(request):
    return render(request, 'expenses/privacy.html')

def terms(request):
    return render(request, 'expenses/terms.html')


