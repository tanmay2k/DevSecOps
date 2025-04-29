from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    PROFILE_TYPES = [
        ('OWNER', 'Account Owner'),
        ('MEMBER', 'Family Member'),
    ]
    
    GENDER_CHOICES = [
        ('MALE', 'Male'),
        ('FEMALE', 'Female'),
        ('OTHER', 'Other'),
        ('PREFER_NOT_TO_SAY', 'Prefer not to say'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_type = models.CharField(max_length=10, choices=PROFILE_TYPES, default='OWNER')
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='family_members', null=True, blank=True)
    relationship = models.CharField(max_length=50, blank=True)
    account_type = models.CharField(max_length=10, default='SINGLE')
    date_of_birth = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=20, choices=GENDER_CHOICES, default='PREFER_NOT_TO_SAY')

    def __str__(self):
        return self.user.username

    def is_owner(self):
        return self.profile_type == 'OWNER'
