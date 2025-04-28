from django.urls import path
from . import views

from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
     path('', views.home, name="home"),
     path('', views.home, name="landing"),  # Ensure the 'landing' name is defined here

    path('expenses', views.index, name="expenses"),
    path('add-expense', views.add_expense, name="add-expenses"),
    path('edit-expense/<int:id>', views.expense_edit, name="expense-edit"),
    path('expense-delete/<int:id>', views.delete_expense, name="expense-delete"),
    path('search-expenses', csrf_exempt(views.search_expenses),
         name="search_expenses"),
    path('expense_category_summary', views.expense_category_summary,
         name="expense_category_summary"),
    path('stats', views.stats_view,
         name="stats"),
    path('set-daily-expense-limit/', views.set_expense_limit, name="set-daily-expense-limit"),
    path('set-limit/', views.set_expense_limit, name='set-expense-limit'),
    path('about/', views.about, name="about"),
    path('contact/', views.contact, name="contact"),
    path('privacy/', views.privacy, name="privacy"),
    path('terms/', views.terms, name="terms"),
]
