from django.urls import path
from . import views

urlpatterns = [
    path('', views.userprofile, name="account"),
    path('addSource/', views.addSource, name="addSource"),
    path('add_family_member/', views.add_family_member, name="add_family_member"),
    path('remove_family_member/<int:member_id>/', views.remove_family_member, name="remove_family_member"),
]
