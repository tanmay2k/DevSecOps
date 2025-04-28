from django import forms 
from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth.models import User
from .models import Profile

class User_Profile(UserChangeForm):
    password = None
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email']

class AccountTypeForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['account_type']
        widgets = {
            'account_type': forms.RadioSelect()
        }

class FamilyMemberProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['relationship']
        widgets = {
            'relationship': forms.TextInput(attrs={
                'class': 'bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5',
                'placeholder': 'e.g. Spouse, Child, Parent'
            })
        }
