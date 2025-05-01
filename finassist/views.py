from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Chat
from expenses.models import Expense
from goals.models import Goal
from userincome.models import UserIncome
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.template.loader import render_to_string
from django.db.models import Sum, Avg
from django.db.models.functions import ExtractMonth
from django.utils import timezone
import datetime
import pandas as pd
import numpy as np
import io
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("LLM_API_KEY"),
)

def get_financial_context(user):
    """Get focused financial context for single goal tracking"""
    today = timezone.now().date()
    three_months_ago = today - datetime.timedelta(days=90)
    
    # Get the user's primary active goal
    primary_goal = Goal.objects.filter(
        owner=user,
        end_date__gt=today  # Only future goals
    ).order_by('end_date').first()  # Get the nearest deadline goal
    
    if not primary_goal:
        return {
            'error': 'no_active_goal',
            'message': 'No active financial goal found. Please set a goal first.'
        }
    
    # Calculate goal metrics
    days_until_deadline = (primary_goal.end_date - today).days
    weeks_remaining = days_until_deadline // 7
    months_remaining = days_until_deadline / 30.44  # Average month length
    
    amount_needed = primary_goal.amount_to_save - primary_goal.current_saved_amount
    weekly_target = amount_needed / weeks_remaining if weeks_remaining > 0 else 0
    monthly_target = amount_needed / months_remaining if months_remaining > 0 else 0
    
    progress_percent = (primary_goal.current_saved_amount / primary_goal.amount_to_save * 100) if primary_goal.amount_to_save > 0 else 0
    
    # Get recent expenses for trend analysis
    recent_expenses = Expense.objects.filter(
        owner=user,
        date__gte=three_months_ago
    ).values('amount', 'date', 'category')
    
    # Get income data
    recent_incomes = UserIncome.objects.filter(
        owner=user,
        date__gte=three_months_ago
    ).values('amount', 'date', 'source', 'is_recurring')
    
    # Calculate monthly averages and trends
    total_expenses = sum(expense['amount'] for expense in recent_expenses)
    total_income = sum(income['amount'] for income in recent_incomes)
    
    avg_monthly_expenses = total_expenses / 3  # 3 months of data
    avg_monthly_income = total_income / 3
    
    # Calculate potential monthly savings based on current patterns
    potential_monthly_savings = avg_monthly_income - avg_monthly_expenses
    
    # Categorize expenses by essential vs non-essential
    expense_categories = {}
    for expense in recent_expenses:
        category = expense['category']
        if category not in expense_categories:
            expense_categories[category] = 0
        expense_categories[category] += float(expense['amount'])
    
    # Sort categories by total amount
    sorted_categories = sorted(
        [(cat, amount) for cat, amount in expense_categories.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Build context focused on goal achievement
    context = {
        'goal': {
            'name': primary_goal.name,
            'target_amount': float(primary_goal.amount_to_save),
            'current_saved': float(primary_goal.current_saved_amount),
            'deadline': primary_goal.end_date.strftime('%Y-%m-%d'),
            'days_remaining': days_until_deadline,
            'weeks_remaining': weeks_remaining,
            'months_remaining': round(months_remaining, 1),
            'progress_percent': round(progress_percent, 2),
            'amount_needed': float(amount_needed),
            'weekly_target': round(float(weekly_target), 2),
            'monthly_target': round(float(monthly_target), 2)
        },
        'financial_capacity': {
            'monthly_income': round(avg_monthly_income, 2),
            'monthly_expenses': round(avg_monthly_expenses, 2),
            'potential_monthly_savings': round(potential_monthly_savings, 2),
            'current_savings_deficit': round(monthly_target - potential_monthly_savings, 2)
        },
        'spending_patterns': {
            'top_expenses': [
                {
                    'category': category,
                    'monthly_average': round(float(amount) / 3, 2)
                }
                for category, amount in sorted_categories[:5]
            ],
            'total_monthly_discretionary': sum(
                amount / 3 for cat, amount in sorted_categories 
                if cat.lower() not in ['rent', 'utilities', 'groceries', 'healthcare']
            )
        }
    }
    
    return context

@csrf_exempt
@login_required
def chatbot_view(request):
    """Handle chat interactions with the LLM"""
    chat_history = Chat.objects.filter(user=request.user).order_by('-timestamp')

    if request.method == 'POST':
        user_message = request.POST.get('message')

        if user_message:
            # Get focused financial context
            context = get_financial_context(request.user)
            
            # Check if there's an active goal
            if 'error' in context:
                assistant_message = (
                    "I notice you don't have an active financial goal set up yet. "
                    "To help you effectively, I need a specific goal with a target amount "
                    "and deadline. Would you like to set one up now?"
                )
                Chat.objects.create(user=request.user, message=user_message, response=assistant_message)
                return JsonResponse({
                    'response': assistant_message,
                    'error': context['error']
                })
            
            # Generate response using LLaMA
            try:
                completion = client.chat.completions.create(
                    extra_body={},
                    model="meta-llama/llama-3.3-70b-instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a goal-centric financial planning assistant AI designed to help users achieve a single, clearly defined financial goal based on their income and expense patterns. You have access to the following data:

- Time series data of the user's income
- Time series data of the user's expenses
- A single financial goal that includes:
  - Target amount
  - Target completion date
  - Current progress
  - Required weekly/monthly savings

Your responsibilities:
1. Track progress toward the financial goal
2. Recommend monthly or weekly saving strategies aligned with the user's income and spending patterns
3. Adjust savings recommendations dynamically based on recent changes in income or expenses
4. Warn the user when their current behavior puts the goal at risk
5. Provide clear, step-by-step guidance to help the user stay on track or catch up

Guidelines:
- Every recommendation must be made with the sole aim of helping the user achieve their specified goal
- Use available data to estimate surplus/deficit and project goal achievement feasibility
- Provide time-based progress updates and required course corrections
- Be supportive but realistic — if the goal is unlikely to be met, suggest alternatives or revised timelines
- Include simple summaries of how much needs to be saved and by when"""
                        },
                        {
                            "role": "user",
                            "content": f"Here is my current financial situation and goal:\n{json.dumps(context, indent=2)}\n\nMy question is: {user_message}"
                        }
                    ]
                )
                assistant_message = completion.choices[0].message.content
            except Exception as e:
                assistant_message = f"I apologize, but I encountered an error processing your request: {str(e)[:100]}..."

            # Save to database
            Chat.objects.create(user=request.user, message=user_message, response=assistant_message)

            # Handle AJAX requests
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                updated_chat_history = Chat.objects.filter(user=request.user).order_by('-timestamp')
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

            return render(request, 'finassist/chatbot.html', {'chat_history': chat_history})

    # Display welcome message with example queries
    context = get_financial_context(request.user)
    welcome_context = {
        'chat_history': chat_history,
        'show_welcome': True,  # Flag to show welcome message in template
        'goal': context.get('goal') if 'error' not in context else None,
        'example_queries': [
            "Am I on track to save $5,000 for my emergency fund by August?",
            "What adjustments should I make to hit my goal faster?",
            "How much can I safely save each week without disrupting essential expenses?",
            "If my rent just increased, how does that affect my goal?",
            "Can you break down what I need to save monthly from now until the deadline?"
        ],
        'capabilities': [
            "Weekly/monthly savings targets",
            "Progress updates",
            "Trade-offs or adjustments if you fall behind",
            "Spending cuts to prioritize your goal"
        ]
    }
    
    return render(request, 'finassist/chatbot.html', welcome_context)
