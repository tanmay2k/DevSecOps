from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name="forecast"),
    path('demographic-analysis/', views.demographic_analysis, name="demographic_analysis"),
    path('payment-insights/', views.payment_method_insights, name="payment_insights"),
    path('transaction-analysis/', views.transaction_category_analysis, name="transaction_analysis"),
    path('demographic-correlation/', views.demographic_correlation_view, name="demographic_correlation"),
    path('api/forecast-data/', views.get_forecast_data, name="get_forecast_data"),
    path('api/demographic-correlation/', views.demographic_correlation_analysis, name="api_demographic_correlation"),
]