from rest_framework import serializers
from expenses.models import Expense

class YourDataSerializer(serializers.Serializer):
    description = serializers.CharField()
    category = serializers.CharField()

class ExpenseSerializer(serializers.ModelSerializer):
    payment_method_display = serializers.CharField(source='get_payment_method_display', read_only=True)
    transaction_category_display = serializers.CharField(source='get_transaction_category_display', read_only=True)
    
    class Meta:
        model = Expense
        fields = [
            'id', 'amount', 'date', 'description', 'category', 
            'is_recurring', 'recurring_end_date', 'spent_by',
            'payment_method', 'payment_method_display',
            'transaction_category', 'transaction_category_display'
        ]
