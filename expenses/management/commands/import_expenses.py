import csv
import io
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from expenses.models import Expense, Category
from django.utils.timezone import now
from datetime import datetime

class Command(BaseCommand):
    help = 'Import expenses from CSV data'

    def add_arguments(self, parser):
        parser.add_argument('--username', type=str, default='mj', help='Username to assign expenses to')
        parser.add_argument('--use-input', action='store_true', help='Use input data instead of file')
        
    def handle(self, *args, **options):
        username = options['username']
        use_input = options.get('use_input', False)
        
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'User {username} does not exist'))
            return
            
        if use_input:
            # Use the CSV data provided in the input
            csv_data = """amount,date,description,category
178.50,2025-01-01,Train ticket to Mumbai,Transportation
45.25,2025-01-01,Breakfast coffee and sandwich,Food & Beverage
245.80,2025-01-02,Weekly groceries at BigBasket,Groceries
98.45,2025-01-02,Electricity bill January,Utilities
1290.00,2025-01-03,Winter boots purchase,Shopping
55.20,2025-01-03,Daily parking fee,Transportation
395.75,2025-01-04,Annual medical checkup,Health
85.30,2025-01-04,Mobile data plan renewal,Utilities
620.40,2025-01-05,Family dinner at Punjabi restaurant,Dining
20.15,2025-01-05,Metro transit card top-up,Transportation
52.80,2025-01-06,Office lunch snacks,Groceries
28500.00,2025-01-06,January apartment rent,Housing
72.50,2025-01-07,Mobile phone bill,Utilities
225.90,2025-01-07,New mystery novel collection,Entertainment
65.30,2025-01-08,Auto ride to meeting,Transportation
115.45,2025-01-08,Fresh produce from farmer's market,Groceries
342.20,2025-01-09,Medication refill,Health
195.75,2025-01-09,Weekend movie tickets,Entertainment
455.30,2025-01-10,Team lunch at office,Food & Beverage
72.45,2025-01-10,Highway toll payment,Transportation
125.60,2025-01-11,Dairy products and bread,Groceries
845.75,2025-01-11,Winter jacket purchase,Shopping
42.30,2025-01-12,Water bill payment,Utilities
250.00,2025-01-12,Haircut and styling,Personal Care
62.50,2025-01-13,Bicycle maintenance,Transportation
205.75,2025-01-13,Seasonal fruits and vegetables,Groceries
520.30,2025-01-14,Dental cleaning appointment,Health
105.90,2025-01-14,Movie streaming annual subscription,Entertainment
415.25,2025-01-15,Weekend brunch with colleagues,Food & Beverage
82.40,2025-01-15,Cab fare for airport drop,Transportation
160.30,2025-01-16,Household cleaning supplies,Household
780.50,2025-01-16,Coffee maker for kitchen,Household
32.45,2025-01-17,Local train fare,Transportation
515.90,2025-01-17,Optometrist consultation,Health
395.80,2025-01-18,Monthly parking pass renewal,Transportation
345.60,2025-01-18,Fuel for car,Transportation
105.40,2025-01-19,Yoga class monthly fee,Fitness
112.30,2025-01-19,Generator fuel,Transportation
50.75,2025-01-20,Metro card recharge,Transportation
475.90,2025-01-20,Internet bill payment,Utilities
340.80,2025-01-21,Grocery shopping for week,Groceries
398.50,2025-01-21,Dinner at Chinese restaurant,Dining
28.45,2025-01-22,Mobile recharge,Utilities
525.75,2025-01-22,Theater tickets for weekend show,Entertainment
445.60,2025-03-31,Designer watch purchase,Shopping"""
            csv_file = io.StringIO(csv_data)
        else:
            # Use a file from the filesystem
            csv_file = open('expense_data.csv', 'r')
        
        # Process the CSV data
        reader = csv.DictReader(csv_file)
        
        # First, ensure all categories exist
        categories = set()
        imported_count = 0
        expense_objects = []
        
        for row in reader:
            categories.add(row['category'])
        
        # Create missing categories
        for category_name in categories:
            Category.objects.get_or_create(name=category_name)
        
        # Reset file pointer for rereading
        if use_input:
            csv_file = io.StringIO(csv_data)
        else:
            csv_file.seek(0)
        
        reader = csv.DictReader(csv_file)
        
        # Now import the expenses
        for row in reader:
            try:
                date = datetime.strptime(row['date'], '%Y-%m-%d').date()
                amount = float(row['amount'])
                description = row['description']
                category = row['category']
                
                expense = Expense(
                    amount=amount,
                    date=date,
                    description=description,
                    category=category,
                    owner=user,
                    is_recurring='NO',
                    spent_by='Self'
                )
                expense_objects.append(expense)
                imported_count += 1
                
                if len(expense_objects) >= 100:
                    Expense.objects.bulk_create(expense_objects)
                    expense_objects = []
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error importing row: {row}, error: {e}'))
        
        # Create any remaining expenses
        if expense_objects:
            Expense.objects.bulk_create(expense_objects)
            
        csv_file.close()
        self.stdout.write(self.style.SUCCESS(f'Successfully imported {imported_count} expenses for user {username}'))
