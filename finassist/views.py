from finassist.utils import DateTimeEncoder
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
from django.db.models import Sum, Avg, Count
from django.db.models.functions import ExtractMonth, TruncMonth
from django.utils import timezone
from decimal import Decimal
import datetime
import pandas as pd
import numpy as np
import io
import json
import os
import logging
import time
import traceback
from openai import OpenAI
from dotenv import load_dotenv
from django.core.serializers.json import DjangoJSONEncoder

# Configure logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Custom JSON Encoder to handle date objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)

# Initialize OpenRouter client for model access
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("LLM_API_KEY"),
)

# Function to get comprehensive context without caching
def get_combined_context(user):
    """
    Get combined context with data sampling for large datasets
    
    Parameters:
    user - The user to get context for
    """
    # Calculate date ranges for efficient querying
    today = timezone.now().date()
    three_months_ago = today - datetime.timedelta(days=90)
    one_year_ago = today - datetime.timedelta(days=365)
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
        date__gte=one_year_ago
    ).values("amount", "date", "description", "category")[:100]  # Limit older data
    
    # Get all active goals
    goals = Goal.objects.filter(
        owner=user
    ).filter(
        end_date__gte=today
    ).values("name", "amount_to_save", "current_saved_amount", "end_date")
    
    # Get recent income data
    incomes = UserIncome.objects.filter(
        owner=user,
        date__gte=three_months_ago
    ).values("amount", "date", "source", "description")
    
    # Get aggregated statistics for better context with fewer data points
    expense_stats = {
        'total_expense': Expense.objects.filter(owner=user).aggregate(total=Sum('amount'))['total'] or 0,
        'avg_monthly': Expense.objects.filter(owner=user, date__gte=one_year_ago)
            .annotate(month=TruncMonth('date'))
            .values('month')
            .annotate(total=Sum('amount'))
            .aggregate(avg=Avg('total'))['avg'] or 0
    }
    
    income_stats = {
        'total_income': UserIncome.objects.filter(owner=user).aggregate(total=Sum('amount'))['total'] or 0,
        'avg_monthly': UserIncome.objects.filter(owner=user, date__gte=one_year_ago)
            .annotate(month=TruncMonth('date'))
            .values('month')
            .annotate(total=Sum('amount'))
            .aggregate(avg=Avg('total'))['avg'] or 0
    }
    
    # Category-wise expense breakdown
    category_expenses = list(Expense.objects.filter(
        owner=user,
        date__gte=three_months_ago
    ).values('category').annotate(total=Sum('amount')))
    
    # Monthly trends (last 12 months)
    monthly_expenses = list(Expense.objects.filter(
        owner=user,
        date__gte=one_year_ago
    ).annotate(month=TruncMonth('date'))
    .values('month')
    .annotate(total=Sum('amount'))
    .order_by('month'))
    
    monthly_income = list(UserIncome.objects.filter(
        owner=user,
        date__gte=one_year_ago
    ).annotate(month=TruncMonth('date'))
    .values('month')
    .annotate(total=Sum('amount'))
    .order_by('month'))
    
    # Calculate savings rate
    savings_rate = 0
    if income_stats['avg_monthly'] > 0:
        savings_rate = max(0, (income_stats['avg_monthly'] - expense_stats['avg_monthly']) / income_stats['avg_monthly'] * 100)
    
    # Generate insights on spending patterns
    spending_insights = []
    
    # Check for unusual spending in last month
    if recent_expenses:
        for category in category_expenses:
            cat_name = category['category']
            recent_cat_total = sum(e['amount'] for e in recent_expenses if e['category'] == cat_name)
            
            # Calculate average for this category over past year
            cat_avg = Expense.objects.filter(
                owner=user,
                date__gte=one_year_ago,
                date__lt=one_month_ago,
                category=cat_name
            ).annotate(month=TruncMonth('date')).values('month').annotate(total=Sum('amount')).aggregate(avg=Avg('total'))['avg'] or 0
            
            if cat_avg > 0 and recent_cat_total > cat_avg * 1.5:
                spending_insights.append({
                    'type': 'increase',
                    'category': cat_name,
                    'amount': recent_cat_total,
                    'average': cat_avg,
                    'percent': round((recent_cat_total - cat_avg) / cat_avg * 100, 1)
                })
            elif cat_avg > 0 and recent_cat_total < cat_avg * 0.5:
                spending_insights.append({
                    'type': 'decrease',
                    'category': cat_name,
                    'amount': recent_cat_total,
                    'average': cat_avg,
                    'percent': round((cat_avg - recent_cat_total) / cat_avg * 100, 1)
                })
    
    # Generate goal insights
    goal_insights = []
    for goal in goals:
        days_until_deadline = (goal['end_date'] - today).days
        if days_until_deadline <= 0:
            continue
            
        amount_needed = goal['amount_to_save'] - goal['current_saved_amount']
        monthly_target = amount_needed / (Decimal(days_until_deadline) / Decimal('30.44')) 
        # Check if monthly target exceeds average monthly savings
        avg_monthly_savings = max(0, income_stats['avg_monthly'] - expense_stats['avg_monthly'])
        
        if monthly_target > avg_monthly_savings:
            goal_insights.append({
                'name': goal['name'],
                'monthly_target': monthly_target,
                'avg_savings': avg_monthly_savings,
                'deficit': monthly_target - avg_monthly_savings,
                'percent_of_income': round(monthly_target / income_stats['avg_monthly'] * 100, 1) if income_stats['avg_monthly'] > 0 else 0
            })
    
    # Combine all expenses with preference for recent data
    expenses_list = list(recent_expenses) + list(older_expenses)
    
    # Build comprehensive context with both detailed and aggregated data
    context = {
        "expenses_sample": expenses_list[:150],  # Limit to reasonable sample
        "goals": list(goals),
        "incomes": list(incomes),
        "stats": {
            "expense_total": expense_stats['total_expense'],
            "expense_average_monthly": expense_stats['avg_monthly'],
            "income_total": income_stats['total_income'],
            "income_average_monthly": income_stats['avg_monthly'],
            "categories": category_expenses,
            "savings_rate": round(savings_rate, 1),
            "monthly_trends": {
                "expenses": monthly_expenses,
                "income": monthly_income
            }
        },
        "insights": {
            "spending": spending_insights,
            "goals": goal_insights,
        },
        "financial_health": {
            "has_emergency_fund": any(g['name'].lower().find('emergency') >= 0 for g in goals),
            "positive_cash_flow": income_stats['avg_monthly'] > expense_stats['avg_monthly'],
            "savings_rate_status": "good" if savings_rate >= 20 else ("moderate" if savings_rate >= 10 else "needs_improvement")
        }
    }
    
    return context

@csrf_exempt
@login_required
def chatbot_view(request):
    """Handle chat interactions with the financial advisor AI"""
    chat_history = Chat.objects.filter(user=request.user).order_by('-timestamp')

    if request.method == 'POST':
        user_message = request.POST.get('message')

        if user_message:
            # Get comprehensive financial context
            financial_context = get_combined_context(request.user)
            
            # Generate response using LLM
            start_time = time.time()  # Moved outside the try block
            request_id = f"req_{int(time.time())}"
            try:
                # Log request details
                logger.info(f"[{request_id}] Starting LLM API call for user: {request.user.username}")
                logger.info(f"[{request_id}] User query: {user_message[:100]}...")
                logger.debug(f"[{request_id}] Financial context size: {len(json.dumps(financial_context, cls=DateTimeEncoder))} bytes")
                
                # Make client call
                completion = client.chat.completions.create(
                    extra_body={},
                    model="meta-llama/llama-3.3-70b-instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a comprehensive and wise financial advisor AI with expertise in personal finance management, budgeting, saving strategies, debt management, and financial planning. Your goal is to provide personalized, actionable financial advice based on the user's actual financial data.

You have access to the following financial data:
1. Income details (sources, amounts, frequency)
2. Expense records (categories, amounts, patterns)
3. Financial goals (targets, deadlines, current progress)
4. Spending patterns and trends
5. Savings rate and financial health metrics

Your approach to financial advice:
1. PERSONALIZED: Always tailor advice to the user's specific financial situation shown in the data
2. EVIDENCE-BASED: Refer to patterns and trends from their actual data to justify recommendations
3. ACTIONABLE: Provide concrete, specific steps the user can take, not just general principles
4. HOLISTIC: Consider multiple aspects of their finances, not just the specific question
5. PRIORITIZED: Focus on the highest-impact changes first
6. REALISTIC: Suggest achievable improvements rather than drastic changes
7. EDUCATIONAL: Explain financial concepts clearly when relevant

When answering questions:
- Analyze the financial data provided to identify relevant patterns or issues
- Connect your advice directly to what you observe in their financial records
- Be specific about amounts, percentages, and timeframes when making recommendations
- Acknowledge both strengths and weaknesses in their financial situation
- When discussing goals, balance optimism with realism about feasibility
- Maintain a supportive, non-judgmental tone even when giving constructive feedback
- End with a simple, high-impact next step the user could take"""
                        },
                        {
                            "role": "user",
                            "content": f"Here is my comprehensive financial data:\n{json.dumps(financial_context, indent=2, cls=DateTimeEncoder)}\n\nMy question is: {user_message}"
                        }
                    ]
                )
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Log successful response details
                assistant_message = completion.choices[0].message.content
                token_count = completion.usage.total_tokens if hasattr(completion, 'usage') else 'Unknown'
                prompt_tokens = completion.usage.prompt_tokens if hasattr(completion.usage, 'prompt_tokens') else 'Unknown'
                completion_tokens = completion.usage.completion_tokens if hasattr(completion.usage, 'completion_tokens') else 'Unknown'
                
                logger.info(f"[{request_id}] LLM API call successful. Response time: {response_time:.2f}s")
                logger.info(f"[{request_id}] Total tokens: {token_count}, Prompt: {prompt_tokens}, Completion: {completion_tokens}")
                logger.debug(f"[{request_id}] Response preview: {assistant_message[:100]}...")
                
            except Exception as e:
                # Enhanced error logging with traceback
                error_time = time.time() - start_time
                logger.error(f"[{request_id}] LLM API call failed after {error_time:.2f}s: {str(e)}")
                logger.error(f"[{request_id}] Error type: {type(e).__name__}")
                logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
                
                assistant_message = f"I apologize, but I encountered an error processing your request. This might be due to connectivity issues or service limitations. Please try again in a moment, or rephrase your question."

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
    welcome_context = {
        'chat_history': chat_history,
        'show_welcome': True,  # Flag to show welcome message in template
        'example_queries': [
            "How am I doing financially overall?",
            "Where am I spending too much money?",
            "How can I improve my savings rate?",
            "Am I on track to meet my financial goals?",
            "What should I prioritize: paying off debt or saving more?",
            "How much should I be setting aside for retirement?",
            "Do my spending patterns align with my financial priorities?",
            "How can I reduce my expenses in the highest spending categories?",
            "What financial habits should I change to improve my situation?",
            "How does my spending compare to typical financial recommendations?"
        ],
        'capabilities': [
            "Personalized budgeting advice",
            "Goal progress tracking",
            "Spending pattern analysis",
            "Savings optimization strategies",
            "Financial health assessment",
            "Debt management guidance",
            "Investment allocation suggestions",
            "Emergency fund planning",
            "Tax efficiency recommendations"
        ]
    }
    
    return render(request, 'finassist/chatbot.html', welcome_context)
