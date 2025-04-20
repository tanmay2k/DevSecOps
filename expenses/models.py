from django.db import models
from django.contrib.auth.models import User
from django.utils.timezone import now
from userprofile.models import Profile

class Expense(models.Model):
    RECURRING_CHOICES = [
        ('NO', 'No'),
        ('DAILY', 'Daily'),
        ('WEEKLY', 'Weekly'),
        ('MONTHLY', 'Monthly'),
        ('YEARLY', 'Yearly'),
    ]

    amount = models.FloatField()
    date = models.DateField(default=now)
    description = models.TextField()
    owner = models.ForeignKey(to=User, on_delete=models.CASCADE)
    category = models.CharField(max_length=266)
    is_recurring = models.CharField(max_length=10, choices=RECURRING_CHOICES, default='NO')
    recurring_end_date = models.DateField(null=True, blank=True)
    spent_by = models.CharField(max_length=50)  # Removed choices to make it dynamic

    def __str__(self):
        return self.category

    class Meta:
        ordering = ['-date']

    @staticmethod
    def get_user_viewable_expenses(user):
        """
        Get all expenses that a user can view based on their profile type
        """
        try:
            profile = user.profile
            if profile.is_owner():
                # Owner can see all expenses from family members
                family_members = User.objects.filter(profile__owner=user)
                return Expense.objects.filter(owner__in=list(family_members) + [user])
            else:
                # Family members can only see their own expenses
                return Expense.objects.filter(owner=user)
        except Profile.DoesNotExist:
            # Fallback to just user's expenses if no profile exists
            return Expense.objects.filter(owner=user)

    @staticmethod
    def get_spent_by_choices(user):
        """
        Get valid spent_by choices for a user based on their profile
        """
        try:
            profile = user.profile
            # Start with the user themselves
            user_full_name = user.get_full_name() or user.username
            choices = [(user.username, f'{user_full_name} (Self)')]
            
            if profile.is_owner():
                # Add family members for owner
                family_members = Profile.objects.filter(owner=user, profile_type='MEMBER')
                for member in family_members:
                    member_name = member.user.get_full_name() or member.user.username
                    choices.append((
                        member.user.username,
                        f'{member_name} ({member.relationship})'
                    ))
            return choices
        except Profile.DoesNotExist:
            return [(user.username, 'Self')]

class Category(models.Model):
    name = models.CharField(max_length=255)

    class Meta:
        verbose_name_plural = 'Categories'

    def __str__(self):
        return self.name

class ExpenseLimit(models.Model):
    owner = models.ForeignKey(to=User, on_delete=models.CASCADE)
    daily_expense_limit = models.IntegerField()

    def __str__(self):
        return f"{self.owner}'s daily limit: {self.daily_expense_limit}"
