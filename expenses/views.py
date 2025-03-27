from django.shortcuts import render, redirect,HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from .models import Category, Expense
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.paginator import Paginator
import json
from django.http import JsonResponse
from userpreferences.models import UserPreference
import datetime
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from django.contrib.sessions.models import Session
from datetime import date
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
import datetime
from .models import ExpenseLimit
from django.core.mail import send_mail
from django.conf import settings
from django.db.models import Sum, Avg
from django.utils import timezone
from datetime import timedelta
import json
from django.db.models import Sum
from django.utils import timezone
from datetime import timedelta
import json
from .models import Expense, Category, ExpenseLimit
from userpreferences.models import UserPreference

data = pd.read_csv('dataset.csv')

# Preprocessing
stop_words = set(stopwords.words('english'))

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
        data = expenses.values()
        return JsonResponse(list(data), safe=False)


@login_required(login_url='/authentication/login')
def index(request):
    categories = Category.objects.all()
    expenses = Expense.objects.filter(owner=request.user)

    sort_order = request.GET.get('sort')

    if sort_order == 'amount_asc':
        expenses = expenses.order_by('amount')
    elif sort_order == 'amount_desc':
        expenses = expenses.order_by('-amount')
    elif sort_order == 'date_asc':
        expenses = expenses.order_by('date')
    elif sort_order == 'date_desc':
        expenses = expenses.order_by('-date')

    paginator = Paginator(expenses, 5)
    page_number = request.GET.get('page')
    page_obj = Paginator.get_page(paginator, page_number)
    try:
        currency = UserPreference.objects.get(user=request.user).currency
    except:
        currency=None

    total = page_obj.paginator.num_pages
    context = {
        'expenses': expenses,
        'page_obj': page_obj,
        'currency': currency,
        'total': total,
        'sort_order': sort_order,

    }
    return render(request, 'expenses/index.html', context)

daily_expense_amounts = {}

def send_limit_notification(user, amount):
    try:
        subject = 'Daily Expense Limit Exceeded'
        message = f'Hello {user.username},\n\nYour expenses for today have exceeded your daily expense limit.'
        send_mail(
            subject, 
            message, 
            settings.EMAIL_HOST_USER, 
            [user.email], 
            fail_silently=True  # Changed to True to prevent validation errors
        )
        return True
    except Exception as e:
        print(f"Email error: {str(e)}")
        return False

@login_required(login_url='/authentication/login')
def add_expense(request):
    categories = Category.objects.all()
    context = {
        'categories': categories,
        'values': request.POST
    }
    
    if request.method == 'POST':
        amount = request.POST.get('amount', '')
        description = request.POST.get('description', '')
        category = request.POST.get('category', '')
        date_str = request.POST.get('expense_date', '')

        print(f"Processing expense with date: {date_str}")

        if not all([amount, description, category, date_str]):
            messages.error(request, 'All fields are required')
            return render(request, 'expenses/add_expense.html', context)

        try:
            # First validate and create the expense
            expense_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            today = datetime.date.today()

            if expense_date > today:
                messages.error(request, 'Date cannot be in the future')
                return render(request, 'expenses/add_expense.html', context)

            # Create the expense first
            expense = Expense.objects.create(
                owner=request.user,
                amount=amount,
                date=expense_date,
                category=category,
                description=description
            )

            # Then check expense limits and send email if needed
            user = request.user
            expense_limits = ExpenseLimit.objects.filter(owner=user)
            if expense_limits.exists():
                daily_limit = expense_limits.first().daily_expense_limit
                total_expenses = get_expense_of_day(user)
                
                if total_expenses > daily_limit:
                    if send_limit_notification(user, amount):
                        messages.warning(request, 'Daily expense limit exceeded. Notification sent.')
                    else:
                        messages.warning(request, 'Daily expense limit exceeded.')

            messages.success(request, 'Expense saved successfully')
            return redirect('expenses')

        except ValueError as e:
            print(f"Date error: {str(e)}")
            messages.error(request, 'Please select a valid date')
            return render(request, 'expenses/add_expense.html', context)
        except Exception as e:
            print(f"General error: {str(e)}")
            messages.error(request, 'An error occurred while saving')
            return render(request, 'expenses/add_expense.html', context)

    return render(request, 'expenses/add_expense.html', context)

@login_required(login_url='/authentication/login')
def expense_edit(request, id):
    expense = Expense.objects.get(pk=id)
    categories = Category.objects.all()
    context = {
        'expense': expense,
        'values': expense,
        'categories': categories
    }
    if request.method == 'GET':
        return render(request, 'expenses/edit-expense.html', context)
    if request.method == 'POST':
        amount = request.POST['amount']
        date_str = request.POST.get('expense_date')

        if not amount:
            messages.error(request, 'Amount is required')
            return render(request, 'expenses/edit-expense.html', context)
        description = request.POST['description']
        date = request.POST['expense_date']
        category = request.POST['category']

        if not description:
            messages.error(request, 'description is required')
            return render(request, 'expenses/edit-expense.html', context)

        try:
            # Convert the date string to a datetime object and validate the date
            date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            today = datetime.date.today()

            if date > today:
                messages.error(request, 'Date cannot be in the future')
                return render(request, 'expenses/add_expense.html', context)

            expense.owner = request.user
            expense.amount = amount
            expense. date = date
            expense.category = category
            expense.description = description

            expense.save()
            messages.success(request, 'Expense saved successfully')

            return redirect('expenses')
        except ValueError:
            messages.error(request, 'Invalid date format')
            return render(request, 'expenses/edit_income.html', context)

        # expense.owner = request.user
        # expense.amount = amount
        # expense. date = date
        # expense.category = category
        # expense.description = description

        # expense.save()

        # messages.success(request, 'Expense updated  successfully')

        # return redirect('expenses')

@login_required(login_url='/authentication/login')
def delete_expense(request, id):
    expense = Expense.objects.get(pk=id)
    expense.delete()
    messages.success(request, 'Expense removed')
    return redirect('expenses')

@login_required(login_url='/authentication/login')
def expense_category_summary(request):
    todays_date = datetime.date.today()
    six_months_ago = todays_date-datetime.timedelta(days=30*6)
    expenses = Expense.objects.filter(owner=request.user,
                                      date__gte=six_months_ago, date__lte=todays_date)
    finalrep = {}

    def get_category(expense):
        return expense.category
    category_list = list(set(map(get_category, expenses)))

    def get_expense_category_amount(category):
        amount = 0
        filtered_by_category = expenses.filter(category=category)

        for item in filtered_by_category:
            amount += item.amount
        return amount

    for x in expenses:
        for y in category_list:
            finalrep[y] = get_expense_category_amount(y)

    return JsonResponse({'expense_category_data': finalrep}, safe=False)

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
    
    # Get top categories (handle empty case)
    top_categories = expenses.values('category')\
        .annotate(total=Sum('amount'))\
        .order_by('-total')[:5]
    
    # Prepare chart data
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
    
    context = {
        'total_expenses': total_expenses,
        'monthly_average': round(monthly_average, 2),
        'top_categories': [
            {'name': cat['category'], 'total': cat['total'] or 0} 
            for cat in top_categories
        ],
        'date_range': date_range,
        'chart_data': json.dumps(chart_data)
    }
    
    return render(request, 'expenses/stats.html', context)

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
    
def get_expense_of_day(user):
    current_date=date.today()
    expenses=Expense.objects.filter(owner=user,date=current_date)
    total_expenses=sum(expense.amount for expense in expenses)
    return total_expenses


