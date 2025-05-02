from django.shortcuts import render, redirect, get_object_or_404
from .models import Goal
from userincome.models import UserIncome
from expenses.models import Expense
from .forms import GoalForm, AddAmountForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.mail import send_mail
from django.db.models import Sum
from django.utils import timezone
import json
from decimal import Decimal
import requests  # Replace OpenAI with standard requests
from django.conf import settings
import logging

import os
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or os.getenv("API_BASE")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL") or os.getenv("MODEL_NAME")


# API KEY configuration for local model (same as finassist)
#client = OpenAI(
#    base_url="https://openrouter.ai/api/v1",
#    api_key=os.getenv("LLM_API_KEY"),
#)

def generate_ai_recommendations(user):
    # Fetch user data
    goals = Goal.objects.filter(owner=user)
    incomes = UserIncome.objects.filter(owner=user)
    expenses = Expense.objects.filter(owner=user)
    
    # Convert Decimal values to floats for JSON serialization
    current_month = timezone.now().month
    monthly_income = float(incomes.filter(date__month=current_month).aggregate(Sum('amount'))['amount__sum'] or 0.0)
    monthly_expenses = float(expenses.filter(date__month=current_month).aggregate(Sum('amount'))['amount__sum'] or 0.0)
    net_cash_flow = monthly_income - monthly_expenses
    
    # Convert Decimal to float in expense analysis
    expense_analysis = expenses.values('category').annotate(total=Sum('amount')).order_by('-total')
    top_expenses = [{
        "category": e['category'], 
        "amount": float(e['total'])
    } for e in expense_analysis[:3]]
    
    # Convert Decimal to float in goals analysis
    goals_analysis = []
    for goal in goals:
        progress = goal.calculate_progress()
        goals_analysis.append({
            "name": goal.name,
            "progress": float(progress['saved_percentage']),
            "daily_required": float(progress['daily_savings_required']),
            "status": "behind" if progress['daily_savings_required'] > Decimal('0') else "on_track"
        })
    
    # Create an optimized prompt for local model (much shorter than before)
    prompt = """You are a financial advisor. Be brief and specific.

Financial Summary:
- Monthly Income: ₹{income:.2f}
- Monthly Expenses: ₹{expenses:.2f}
- Net Cash Flow: ₹{cashflow:.2f}

Top Expenses:
{top_expenses}

Goals:
{goals}

Give 3-5 actionable recommendations to improve finances. Be specific.""".format(
        income=monthly_income,
        expenses=monthly_expenses,
        cashflow=net_cash_flow,
        top_expenses="\n".join([f"- {e['category']}: ₹{e['amount']:.2f}" for e in top_expenses[:2]]),
        goals="\n".join([f"- {g['name']}: {g['progress']:.1f}% complete" for g in goals_analysis[:2]])
    )
    
    # Call the Ollama API with increased timeout and retry logic
    max_retries = 2
    current_retry = 0
    timeout_seconds = 45  # Increased timeout for slower systems
    
    while current_retry <= max_retries:
        try:
            # Using the generate endpoint with minimal parameters
            response = requests.post(
                f"{OLLAMA_BASE_URL}/generate",
                headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY')}"},
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,  # Lower temperature for more predictable responses
                        "num_predict": 500,  # Limit token count for faster generation
                        "top_k": 40,
                        "top_p": 0.9
                    }
                },
                timeout=timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                recommendations = result.get("response", "")
                
                # Format the response with headers if successful
                if recommendations:
                    formatted_response = "## Financial Recommendations\n\n" + recommendations
                    return formatted_response
                else:
                    return "Based on your financial data, I recommend tracking expenses more carefully and setting aside a portion of your income for savings goals."
            else:
                current_retry += 1
                if current_retry > max_retries:
                    logging.error(f"Failed to generate recommendations after {max_retries} attempts")
                    return "I'm unable to generate personalized recommendations right now. As a general tip, try to save at least 20% of your income and reduce spending in your highest expense categories."
        
        except requests.exceptions.Timeout:
            current_retry += 1
            timeout_seconds += 15  # Increase timeout for next retry
            if current_retry > max_retries:
                logging.error("Timeout while generating AI recommendations")
                return "I apologize, but generating detailed recommendations is taking too long. A good financial practice is to maintain an emergency fund and review your largest expense categories for potential savings."
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            return "I encountered an issue analyzing your financial data. Consider the 50/30/20 rule: 50% for needs, 30% for wants, and 20% for savings and debt repayment."
    
    return "Unable to provide personalized recommendations at this time. Please try again later."

@login_required
def add_goal(request):
    if request.method == 'POST':
        form = GoalForm(request.POST)
        if form.is_valid():
            goal = form.save(commit=False)  # Delay saving to add owner
            goal.owner = request.user
            goal.save()
            return redirect('list_goals')
        else:
            # If form is invalid, re-render with error messages
            return render(request, 'goals/add_goals.html', {'form': form})
    else:
        # For non-POST requests, show the form (optional)
        form = GoalForm()
        return render(request, 'goals/add_goals.html', {'form': form})

@login_required(login_url='/authentication/login')
def list_goals(request):

    # goals = Goal.objects.all()
    goals = Goal.objects.filter(owner=request.user)
    ai_recommendations = generate_ai_recommendations(request.user)
    
    context = {
        'goals': goals,
        'ai_recommendations': ai_recommendations
    }
    add_amount_form = AddAmountForm() 
    return render(request, 'goals/list_goals.html', context)


@login_required(login_url='/authentication/login')
def add_amount(request, goal_id):
    goal = get_object_or_404(Goal, pk=goal_id)

    if request.method == 'POST':
        form = AddAmountForm(request.POST)
        if form.is_valid():
            additional_amount = form.cleaned_data['additional_amount']
            amount_required = goal.amount_to_save - goal.current_saved_amount

            if additional_amount > amount_required:
                messages.error(request, f'The maximum amount needed to achieve goal is : {amount_required}.')
            else:
                goal.current_saved_amount += additional_amount
                goal.save()

                # Check if the goal is achieved
                if goal.current_saved_amount == goal.amount_to_save:
                    # Send congratulatory email to the user
                        
                        send_congratulatory_email(request.user.email, goal)
                        messages.success(request, 'Congratulations! You have achieved your goal.')

                        # Disable the "Add Amount" button
                        goal.is_achieved = True
                        goal.delete()
               
                else:
                    messages.success(request, f'Amount successfully added. Total saved amount: {goal.current_saved_amount}.')
                    messages.info(request, f'Amount required to reach goal: {amount_required}.')

        return redirect('list_goals')

    # Redirect to list_goals if the request method is not POST
    return redirect('list_goals')

def send_congratulatory_email(email, goal):
    subject = 'Congratulations on achieving your goal!'
    message = f'Dear User,\n\nCongratulations on achieving your goal "{goal.name}". You have successfully saved {goal.amount_to_save}.\n\nKeep up the good work!\n\nBest regards,\nThe Goal Tracker Team, \nWealthWizard Team'
    send_mail(subject, message, '<your email>', [email])
    
    



def delete_goal(request, goal_id):
    try:
        goal = Goal.objects.get(id=goal_id,owner=request.user)
        goal.delete()
        return redirect('list_goals')
    except Goal.DoesNotExist:
        pass
