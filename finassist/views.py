from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Chat
from expenses.models import Expense
from goals.models import Goal
from userincome.models import UserIncome
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
import requests  # Add requests for Ollama API calls
from django.template.loader import render_to_string
from django.db.models import Sum, Avg
from django.utils import timezone
import datetime
from django.core.cache import cache
import pandas as pd
import numpy as np
import io
import json

# Replace OpenAI client with Ollama configuration
OLLAMA_BASE_URL = "ollama-service:11434/api"
OLLAMA_MODEL = "gemma:2b"

# Function to get additional context with optimizations for large datasets
def get_combined_context(user, cache_timeout=300):
    """
    Get combined context with caching and data sampling for large datasets
    
    Parameters:
    user - The user to get context for
    cache_timeout - Cache timeout in seconds (default: 5 minutes)
    """
    # Generate a unique cache key for this user
    cache_key = f"user_financial_context_{user.id}"
    
    # Try to get cached context first
    cached_context = cache.get(cache_key)
    if cached_context:
        return cached_context
    
    # Calculate date ranges for efficient querying
    today = timezone.now().date()
    three_months_ago = today - datetime.timedelta(days=90)
    one_month_ago = today - datetime.timedelta(days=30)
    
    # Get recent and relevant data with optimized queries
    # Recent expenses (last 30 days) + sample of older expenses (last 90 days)
    recent_expenses = Expense.objects.filter(
        owner=user, 
        date__gte=one_month_ago
    ).values("amount", "date", "description", "category")
    
    older_expenses = Expense.objects.filter(
        owner=user, 
        date__lt=one_month_ago,
        date__gte=three_months_ago
    ).values("amount", "date", "description", "category")[:100]  # Limit older data
    
    # Get only active goals or recently completed ones
    goals = Goal.objects.filter(
        owner=user
    ).filter(
        end_date__gte=three_months_ago
    ).values("name", "amount_to_save", "current_saved_amount", "end_date")
    
    # Get recent income data
    incomes = UserIncome.objects.filter(
        owner=user,
        date__gte=three_months_ago
    ).values("amount", "date", "source", "description")
    
    # Get aggregated statistics for better context with fewer data points
    expense_stats = Expense.objects.filter(owner=user).aggregate(
        total_expense=Sum('amount'),
        avg_monthly=Avg('amount')
    )
    
    income_stats = UserIncome.objects.filter(owner=user).aggregate(
        total_income=Sum('amount'),
        avg_monthly=Avg('amount')
    )
    
    # Category-wise expense breakdown
    category_expenses = list(Expense.objects.filter(
        owner=user,
        date__gte=three_months_ago
    ).values('category').annotate(total=Sum('amount')))
    
    # Combine all expenses with preference for recent data
    expenses_list = list(recent_expenses) + list(older_expenses)
    
    # Build optimized context with both detailed and aggregated data
    context = {
        "recent_expenses": list(recent_expenses),
        "expenses_sample": expenses_list[:200],  # Limit to reasonable sample
        "goals": list(goals),
        "incomes": list(incomes),
        "stats": {
            "expense_total": expense_stats.get('total_expense', 0),
            "expense_average": expense_stats.get('avg_monthly', 0),
            "income_total": income_stats.get('total_income', 0),
            "income_average": income_stats.get('avg_monthly', 0),
            "categories": category_expenses
        },
        "seasonal_trends": generate_seasonal_trends(user),
    }
    
    # Cache the context
    cache.set(cache_key, context, cache_timeout)
    
    return context

# Generate seasonal trends and patterns for better predictions
def generate_seasonal_trends(user):
    """Generate seasonal patterns and trends from user expense data"""
    try:
        # Get expenses from the last year
        one_year_ago = timezone.now().date() - datetime.timedelta(days=365)
        expenses = Expense.objects.filter(
            owner=user,
            date__gte=one_year_ago
        ).values('amount', 'date', 'category')
        
        if not expenses:
            return {"message": "Not enough data for seasonal analysis"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(list(expenses))
        
        # Generate monthly aggregation
        if not df.empty and 'date' in df.columns and 'amount' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            monthly_totals = df.groupby('month')['amount'].sum().to_dict()
            
            # Generate category-wise monthly trends
            category_trends = {}
            if 'category' in df.columns:
                for category in df['category'].unique():
                    cat_data = df[df['category'] == category]
                    if not cat_data.empty:
                        category_trends[category] = cat_data.groupby('month')['amount'].sum().to_dict()
            
            return {
                "monthly_spending": monthly_totals,
                "category_trends": category_trends
            }
        
        return {"message": "Data format unsuitable for trend analysis"}
        
    except Exception as e:
        return {"error": str(e)}

# Function to generate chatbot response using local Ollama model
def generate_response(user_input, context):
    """Generate response from local Ollama LLM with optimized context size for Gemma 2B"""
    # For Gemma 2B on an i3, we need to be very careful with context size
    
    # Further reduce context for performance
    query_keywords = user_input.lower()
    
    # Create a very minimal prompt for Gemma 2B
    prompt = "You are a financial assistant. Answer briefly.\n\n"
    
    # Only add the most essential context based on query
    if any(word in query_keywords for word in ['expense', 'spend', 'cost']):
        # Just summarize expenses in one line
        total = context.get('stats', {}).get('expense_total', 0)
        prompt += f"Total expenses: ₹{total}. "
        
        # Add at most 1 recent expense as example
        recent = context.get('recent_expenses', [])
        if recent and len(recent) > 0:
            exp = recent[0]
            prompt += f"Most recent: {exp.get('category')} ₹{exp.get('amount')}. "
    
    if any(word in query_keywords for word in ['goal', 'save']):
        # Only include the first goal if any
        goals = context.get('goals', [])
        if goals and len(goals) > 0:
            goal = goals[0]
            prompt += f"Goal: {goal.get('name')} - Progress: ₹{goal.get('current_saved_amount')}/₹{goal.get('amount_to_save')}. "
    
    if any(word in query_keywords for word in ['income', 'earn']):
        # Just the total income
        total = context.get('stats', {}).get('income_total', 0)
        prompt += f"Total income: ₹{total}. "
    
    # Add the actual user question
    prompt += f"\nUser: {user_input}\n\nAssistant:"
    
    # Call the Ollama API with increased timeout and retry logic
    max_retries = 2
    current_retry = 0
    timeout_seconds = 45  # Increased timeout for slower systems
    
    while current_retry <= max_retries:
        try:
            # Using the completion endpoint with minimal parameters
            response = requests.post(
                f"{OLLAMA_BASE_URL}/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,  # Lower temperature for more predictable responses
                        "num_predict": 150,  # Even fewer tokens for faster generation
                        "top_k": 40,
                        "top_p": 0.9
                    }
                },
                timeout=timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "I've analyzed your financial data and can provide a brief insight.")
            else:
                current_retry += 1
                if current_retry > max_retries:
                    return "Sorry, I'm having trouble processing your request right now. Please try a shorter question."
        
        except requests.exceptions.Timeout:
            current_retry += 1
            timeout_seconds += 15  # Increase timeout for next retry
            if current_retry > max_retries:
                return "I apologize, but the response is taking too long. Try asking a simpler question or check again later."
            
        except Exception as e:
            return f"I encountered an issue while processing your request. Error: {str(e)[:50]}..."
    
    return "Unable to get a response at this time. Please try again later."

@csrf_exempt
@login_required
def chatbot_view(request):
    chat_history = Chat.objects.filter(user=request.user).order_by('-timestamp')

    if request.method == 'POST':
        user_message = request.POST.get('message')

        if user_message:
            # Use the optimized context gathering function
            context = get_combined_context(request.user)

            # Generate the assistant's response
            assistant_message = generate_response(user_message, context)

            # Save to database
            Chat.objects.create(user=request.user, message=user_message, response=assistant_message)

            # For AJAX requests, return JSON response
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                # Get updated chat history
                updated_chat_history = Chat.objects.filter(user=request.user).order_by('-timestamp')
                
                # Render the updated chat history to HTML
                chat_history_html = render_to_string(
                    'finassist/partials/chat_history.html',
                    {'chat_history': updated_chat_history},
                    request=request
                )

                return JsonResponse({
                    'response': assistant_message,
                    'updated_data': {
                        'chat_history': chat_history_html
                    }
                })

            # For regular requests, return the full page
            return render(request, 'finassist/chatbot.html', {'chat_history': chat_history})

    # GET request - render the initial page
    return render(request, 'finassist/chatbot.html', {'chat_history': chat_history})

@login_required
def test_ollama_connection(request):
    """Test if Ollama is running properly with the Gemma model"""
    try:
        # Increase timeout for model response check
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=30)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            gemma_available = any("gemma:2b" in model.get("name", "") for model in models)
            
            if gemma_available:
                # Try a simple prompt with increased timeout
                test_response = requests.post(
                    f"{OLLAMA_BASE_URL}/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": "Hello",  # Even shorter prompt for testing
                        "stream": False,
                        "options": {
                            "num_predict": 20,  # Minimal response for testing
                            "temperature": 0.1  # Low temperature for faster response
                        }
                    },
                    timeout=45  # Much longer timeout for i3 system
                )
                
                if test_response.status_code == 200:
                    return JsonResponse({
                        "status": "success", 
                        "message": "Ollama with Gemma 2B is running properly",
                        "sample_response": test_response.json().get("response")
                    })
                else:
                    return JsonResponse({
                        "status": "error",
                        "message": f"Gemma model loaded but generation failed with status code: {test_response.status_code}"
                    })
            else:
                return JsonResponse({
                    "status": "error",
                    "message": "Gemma 2B model not found. Please run 'ollama pull gemma:2b' first.",
                    "available_models": [model.get("name") for model in models]
                })
        else:
            return JsonResponse({
                "status": "error",
                "message": f"Ollama returned status code: {response.status_code}"
            })
    except requests.exceptions.Timeout:
        return JsonResponse({
            "status": "error",
            "message": "Connection to Ollama timed out. This is common on i3 systems. Try restarting Ollama with 'ollama serve --gpu 0' for CPU-only mode."
        })
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"Error connecting to Ollama: {str(e)}"
        })
