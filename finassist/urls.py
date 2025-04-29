from django.urls import path
from . import views

from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
    path('', views.chatbot_view, name="chatbot"),
    path('test-ollama/', views.test_ollama_connection, name="test_ollama"),
]