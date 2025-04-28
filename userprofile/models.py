from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    ACCOUNT_TYPE_CHOICES = [
        ('SOLO', 'Solo Account'),
        ('MULTI', 'Multi-User Account'),
    ]

    PROFILE_TYPE_CHOICES = [
        ('OWNER', 'Account Owner'),
        ('MEMBER', 'Family Member'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    account_type = models.CharField(max_length=5, choices=ACCOUNT_TYPE_CHOICES, default='SOLO')
    profile_type = models.CharField(max_length=6, choices=PROFILE_TYPE_CHOICES, default='OWNER')
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='family_members', null=True, blank=True)
    relationship = models.CharField(max_length=50, blank=True, null=True, help_text='Relationship to account owner')
    
    def __str__(self):
        if self.profile_type == 'OWNER':
            return f"{self.user.username}'s Profile (Owner)"
        return f"{self.user.username}'s Profile ({self.relationship})"

    def is_owner(self):
        return self.profile_type == 'OWNER'

    def can_view_all_data(self):
        return self.profile_type == 'OWNER'
