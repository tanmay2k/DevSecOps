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
            # Use the CSV data provided in the input with seasonality patterns
            csv_data = """amount,date,description,category,payment_method,transaction_category
178.50,2025-01-01,Train ticket to Mumbai,Transportation,CASH,OFFLINE_MARKET
45.25,2025-01-01,Breakfast coffee and sandwich,Food & Beverage,CASH,RESTAURANT
245.80,2025-01-02,Weekly groceries at BigBasket,Groceries,CREDIT_CARD,ECOMMERCE
98.45,2025-01-02,Electricity bill January,Utilities,NET_BANKING,UTILITY
1290.00,2025-01-03,Winter boots purchase,Shopping,CREDIT_CARD,MALL
55.20,2025-01-03,Daily parking fee,Transportation,CASH,OFFLINE_MARKET
395.75,2025-01-04,Annual medical checkup,Health,DEBIT_CARD,OFFLINE_MARKET
85.30,2025-01-04,Mobile data plan renewal,Utilities,UPI,UTILITY
620.40,2025-01-05,Family dinner at Punjabi restaurant,Dining,CREDIT_CARD,RESTAURANT
20.15,2025-01-05,Metro transit card top-up,Transportation,CASH,OFFLINE_MARKET
52.80,2025-01-06,Office lunch snacks,Groceries,UPI,QUICK_COMMERCE
28500.00,2025-01-06,January apartment rent,Housing,NET_BANKING,UTILITY
72.50,2025-01-07,Mobile phone bill,Utilities,UPI,UTILITY
225.90,2025-01-07,New mystery novel collection,Entertainment,CREDIT_CARD,ECOMMERCE
65.30,2025-01-08,Auto ride to meeting,Transportation,CASH,OFFLINE_MARKET
115.45,2025-01-08,Fresh produce from farmer's market,Groceries,UPI,QUICK_COMMERCE
342.20,2025-01-09,Medication refill,Health,DEBIT_CARD,OFFLINE_MARKET
195.75,2025-01-09,Weekend movie tickets,Entertainment,CASH,RESTAURANT
455.30,2025-01-10,Team lunch at office,Food & Beverage,CREDIT_CARD,RESTAURANT
72.45,2025-01-10,Highway toll payment,Transportation,CASH,OFFLINE_MARKET
125.60,2025-01-11,Dairy products and bread,Groceries,UPI,QUICK_COMMERCE
845.75,2025-01-11,Winter jacket purchase,Shopping,CREDIT_CARD,MALL
42.30,2025-01-12,Water bill payment,Utilities,NET_BANKING,UTILITY
250.00,2025-01-12,Haircut and styling,Personal Care,CASH,OFFLINE_MARKET
62.50,2025-01-13,Bicycle maintenance,Transportation,CASH,OFFLINE_MARKET
205.75,2025-01-13,Seasonal fruits and vegetables,Groceries,UPI,QUICK_COMMERCE
520.30,2025-01-14,Dental cleaning appointment,Health,DEBIT_CARD,OFFLINE_MARKET
105.90,2025-01-14,Movie streaming annual subscription,Entertainment,CREDIT_CARD,ECOMMERCE
415.25,2025-01-15,Weekend brunch with colleagues,Food & Beverage,CASH,RESTAURANT
82.40,2025-01-15,Cab fare for airport drop,Transportation,CASH,OFFLINE_MARKET
160.30,2025-01-16,Household cleaning supplies,Household,UPI,QUICK_COMMERCE
780.50,2025-01-16,Coffee maker for kitchen,Household,CREDIT_CARD,MALL
32.45,2025-01-17,Local train fare,Transportation,CASH,OFFLINE_MARKET
515.90,2025-01-17,Optometrist consultation,Health,DEBIT_CARD,OFFLINE_MARKET
395.80,2025-01-18,Monthly parking pass renewal,Transportation,CASH,OFFLINE_MARKET
345.60,2025-01-18,Fuel for car,Transportation,CASH,OFFLINE_MARKET
105.40,2025-01-19,Yoga class monthly fee,Fitness,CREDIT_CARD,ECOMMERCE
112.30,2025-01-19,Generator fuel,Transportation,CASH,OFFLINE_MARKET
50.75,2025-01-20,Metro card recharge,Transportation,CASH,OFFLINE_MARKET
475.90,2025-01-20,Internet bill payment,Utilities,UPI,UTILITY
340.80,2025-01-21,Grocery shopping for week,Groceries,UPI,QUICK_COMMERCE
398.50,2025-01-21,Dinner at Chinese restaurant,Dining,CREDIT_CARD,RESTAURANT
28.45,2025-01-22,Mobile recharge,Utilities,CASH,OFFLINE_MARKET
525.75,2025-01-22,Theater tickets for weekend show,Entertainment,CREDIT_CARD,ECOMMERCE
455.30,2025-01-23,Multiplex premium screening tickets,Entertainment,CREDIT_CARD,ECOMMERCE
135.60,2025-01-23,Car refueling,Transportation,CASH,OFFLINE_MARKET
1250.80,2025-01-24,Car insurance quarterly payment,Transportation,CREDIT_CARD,ECOMMERCE
115.25,2025-01-24,February rent advance,Housing,NET_BANKING,UTILITY
185.40,2025-01-25,Motorcycle servicing,Transportation,CASH,OFFLINE_MARKET
295.80,2025-01-25,Prescription medications,Health,DEBIT_CARD,OFFLINE_MARKET
245.75,2025-01-26,Monthly train pass,Transportation,CASH,OFFLINE_MARKET
175.30,2025-01-26,Digital magazine subscription,Entertainment,CREDIT_CARD,ECOMMERCE
350.60,2025-01-27,New formal pants for office,Shopping,CREDIT_CARD,MALL
95.40,2025-01-27,Car wash service,Transportation,CASH,OFFLINE_MARKET
185.30,2025-01-28,Dinner with friends,Dining,CASH,RESTAURANT
215.75,2025-01-28,New shirt purchase,Shopping,CREDIT_CARD,MALL
265.40,2025-01-29,Train ticket to Hyderabad,Transportation,CASH,OFFLINE_MARKET
435.90,2025-01-29,Designer sunglasses,Shopping,CREDIT_CARD,MALL
135.60,2025-01-30,Casual dining experience,Dining,CASH,RESTAURANT
295.40,2025-01-30,Monthly grocery stock-up,Groceries,UPI,QUICK_COMMERCE
125.30,2025-01-31,Coffee shop work session,Food & Beverage,CASH,OFFLINE_MARKET
42.90,2025-01-31,Scarf purchase,Shopping,CREDIT_CARD,MALL
335.60,2025-02-01,Internet bill for February,Utilities,UPI,UTILITY
115.75,2025-02-01,Comedy show tickets,Entertainment,CREDIT_CARD,ECOMMERCE
1580.00,2025-02-02,Traditional outfit purchase,Shopping,CREDIT_CARD,MALL
520.80,2025-02-02,Anniversary dinner at fine dining,Dining,CREDIT_CARD,RESTAURANT
515.30,2025-02-03,Airport parking for weekend trip,Transportation,CASH,OFFLINE_MARKET
1250.75,2025-02-03,Flight tickets for weekend getaway,Transportation,CREDIT_CARD,ECOMMERCE
185.40,2025-02-04,Mobile data plan renewal,Utilities,UPI,UTILITY
72.50,2025-02-04,Evening cafe visit,Food & Beverage,CASH,OFFLINE_MARKET
385.90,2025-02-05,Car service and oil change,Transportation,CASH,OFFLINE_MARKET
265.40,2025-02-05,Monthly gym membership renewal,Fitness,CREDIT_CARD,ECOMMERCE
88.75,2025-02-06,Scooter maintenance,Transportation,CASH,OFFLINE_MARKET
28500.00,2025-02-06,February apartment rent,Housing,NET_BANKING,UTILITY
38.90,2025-02-07,Comic book purchase,Entertainment,CASH,OFFLINE_MARKET
495.80,2025-02-07,Business dinner meeting,Dining,CREDIT_CARD,RESTAURANT
165.30,2025-02-08,Food delivery order,Dining,CASH,RESTAURANT
372.45,2025-02-08,Society maintenance payment,Housing,UPI,UTILITY
195.60,2025-02-09,Train ticket to Pune,Transportation,CASH,OFFLINE_MARKET
295.30,2025-02-09,Airport shuttle service,Transportation,CASH,OFFLINE_MARKET
315.80,2025-02-10,Electricity bill payment,Utilities,UPI,UTILITY
125.40,2025-02-10,Lunch with client,Dining,CASH,RESTAURANT
525.90,2025-02-11,Annual sports club membership,Fitness,CREDIT_CARD,ECOMMERCE
420.75,2025-02-11,March rent advance payment,Housing,NET_BANKING,UTILITY
505.30,2025-02-12,Collector's edition book set,Entertainment,CREDIT_CARD,ECOMMERCE
485.75,2025-02-12,Business attire purchase,Shopping,CREDIT_CARD,MALL
345.80,2025-02-13,Family dinner outing,Dining,CASH,RESTAURANT
495.40,2025-02-13,Weekly grocery shopping,Groceries,UPI,QUICK_COMMERCE
68.90,2025-02-14,Shopping mall parking,Transportation,CASH,OFFLINE_MARKET
1425.60,2025-02-14,Valentine's Day special dinner,Dining,CREDIT_CARD,RESTAURANT
1245.30,2025-02-14,Valentine's Day gift package,Shopping,CREDIT_CARD,MALL
195.80,2025-02-15,Pilates monthly subscription,Fitness,CREDIT_CARD,ECOMMERCE
235.60,2025-02-16,General physician consultation,Health,DEBIT_CARD,OFFLINE_MARKET
165.40,2025-02-16,Medical test fees,Health,DEBIT_CARD,OFFLINE_MARKET
450.75,2025-02-17,Weekend family brunch,Dining,CASH,RESTAURANT
215.80,2025-02-17,March month rental deposit,Housing,NET_BANKING,UTILITY
175.40,2025-02-18,Spinning class package,Fitness,CREDIT_CARD,ECOMMERCE
305.25,2025-02-18,Home broadband bill payment,Utilities,UPI,UTILITY
98.75,2025-02-19,History book purchase,Entertainment,CASH,OFFLINE_MARKET
440.80,2025-02-19,Petrol fill-up,Transportation,CASH,OFFLINE_MARKET
65.30,2025-02-20,Accessories shopping,Shopping,CREDIT_CARD,MALL
545.90,2025-02-20,Business networking dinner,Dining,CREDIT_CARD,RESTAURANT
425.75,2025-02-21,Business lunch meeting,Dining,CASH,RESTAURANT
135.60,2025-02-21,Health supplements,Health,DEBIT_CARD,OFFLINE_MARKET
22.50,2025-02-22,Afternoon tea break,Food & Beverage,CASH,OFFLINE_MARKET
445.90,2025-02-22,Bus tickets for family trip,Transportation,CASH,OFFLINE_MARKET
395.80,2025-02-23,Scooter fuel fill-up,Transportation,CASH,OFFLINE_MARKET
405.60,2025-02-23,Weekly grocery shopping,Groceries,UPI,QUICK_COMMERCE
420.75,2025-02-24,Electric bill payment,Utilities,UPI,UTILITY
65.40,2025-02-24,Takeout dinner,Dining,CASH,RESTAURANT
215.80,2025-02-25,Movie rental subscription,Entertainment,CASH,OFFLINE_MARKET
85.60,2025-02-25,Vehicle cleaning service,Transportation,CASH,OFFLINE_MARKET
475.90,2025-02-26,Special occasion dinner,Dining,CREDIT_CARD,RESTAURANT
28500.00,2025-02-26,March apartment rent,Housing,NET_BANKING,UTILITY
195.80,2025-02-27,Fresh groceries shopping,Groceries,UPI,QUICK_COMMERCE
58.40,2025-02-27,Street food festival visit,Food & Beverage,CASH,OFFLINE_MARKET
185.90,2025-02-28,Leather handbag purchase,Shopping,CREDIT_CARD,MALL
195.40,2025-02-28,Skin specialist consultation,Health,DEBIT_CARD,OFFLINE_MARKET
395.80,2025-03-01,Airport parking fee,Transportation,CASH,OFFLINE_MARKET
355.60,2025-03-01,April month rent advance,Housing,NET_BANKING,UTILITY
485.90,2025-03-02,Family Sunday brunch,Dining,CASH,RESTAURANT
275.60,2025-03-02,Dance class monthly fee,Fitness,CREDIT_CARD,ECOMMERCE
185.30,2025-03-03,Train ticket to Kolkata,Transportation,CASH,OFFLINE_MARKET
48.90,2025-03-03,Morning coffee and pastry,Food & Beverage,CASH,OFFLINE_MARKET
265.40,2025-03-04,Grocery shopping at D-Mart,Groceries,UPI,QUICK_COMMERCE
142.80,2025-03-04,March electricity bill (increased due to spring),Utilities,UPI,UTILITY
1680.50,2025-03-05,Spring collection dress,Shopping,CREDIT_CARD,MALL
58.90,2025-03-05,Daily office parking,Transportation,CASH,OFFLINE_MARKET
415.75,2025-03-06,Annual medical insurance co-pay,Health,DEBIT_CARD,OFFLINE_MARKET
88.60,2025-03-06,Mobile internet plan,Utilities,UPI,UTILITY
645.30,2025-03-07,Anniversary dinner celebration,Dining,CREDIT_CARD,RESTAURANT
22.50,2025-03-07,Local bus day pass,Transportation,CASH,OFFLINE_MARKET
55.80,2025-03-08,Office break snacks,Groceries,CASH,OFFLINE_MARKET
29000.00,2025-03-08,March month apartment rent,Housing,NET_BANKING,UTILITY
75.40,2025-03-09,Mobile phone monthly bill,Utilities,UPI,UTILITY
235.60,2025-03-09,Biography book purchase,Entertainment,CASH,OFFLINE_MARKET
68.90,2025-03-10,Shared auto ride,Transportation,CASH,OFFLINE_MARKET
125.40,2025-03-10,Vegetables and fruits,Groceries,UPI,QUICK_COMMERCE
355.80,2025-03-11,Pharmacy purchase,Health,DEBIT_CARD,OFFLINE_MARKET
205.40,2025-03-11,Concert tickets,Entertainment,CREDIT_CARD,ECOMMERCE
525.90,2025-03-12,Office team lunch (increased for special event),Food & Beverage,CASH,RESTAURANT
75.60,2025-03-12,Highway toll charges,Transportation,CASH,OFFLINE_MARKET
135.80,2025-03-13,Weekly bread and dairy,Groceries,UPI,QUICK_COMMERCE
1265.30,2025-03-13,Spring wardrobe update (seasonal increase),Shopping,CREDIT_CARD,MALL
45.60,2025-03-14,Water utility bill,Utilities,UPI,UTILITY
295.40,2025-03-14,Spa treatment,Personal Care,CASH,OFFLINE_MARKET
65.30,2025-03-15,Electric scooter maintenance,Transportation,CASH,OFFLINE_MARKET
315.40,2025-03-15,Seasonal fruits shopping (spring produce increase),Groceries,UPI,QUICK_COMMERCE
535.80,2025-03-16,ENT specialist consultation,Health,DEBIT_CARD,OFFLINE_MARKET
115.40,2025-03-16,Music subscription annual plan,Entertainment,CREDIT_CARD,ECOMMERCE
525.90,2025-03-17,St. Patrick's Day celebration,Food & Beverage,CASH,RESTAURANT
85.60,2025-03-17,Rideshare to airport,Transportation,CASH,OFFLINE_MARKET
165.40,2025-03-18,Home cleaning supplies,Household,UPI,QUICK_COMMERCE
895.80,2025-03-18,Spring cleaning service,Household,CREDIT_CARD,ECOMMERCE
35.60,2025-03-19,Local train tickets,Transportation,CASH,OFFLINE_MARKET
525.40,2025-03-19,Dental check-up and cleaning,Health,DEBIT_CARD,OFFLINE_MARKET
405.90,2025-03-20,Monthly parking facility fee,Transportation,CASH,OFFLINE_MARKET
465.40,2025-03-20,Car fuel fill-up (seasonal travel increase),Transportation,CASH,OFFLINE_MARKET
215.80,2025-03-21,Fitness class package (spring fitness resolution),Fitness,CREDIT_CARD,ECOMMERCE
118.90,2025-03-21,Diesel purchase for generator,Transportation,CASH,OFFLINE_MARKET
52.40,2025-03-22,Metro card top-up,Transportation,CASH,OFFLINE_MARKET
485.60,2025-03-22,Home internet monthly bill,Utilities,UPI,UTILITY
455.40,2025-03-23,Weekly grocery shopping (increased for spring parties),Groceries,UPI,QUICK_COMMERCE
515.80,2025-03-23,Dinner at new fusion restaurant,Dining,CREDIT_CARD,RESTAURANT
32.50,2025-03-24,Prepaid mobile recharge,Utilities,CASH,OFFLINE_MARKET
635.60,2025-03-24,Theater festival tickets (spring cultural events),Entertainment,CREDIT_CARD,ECOMMERCE
575.40,2025-03-25,3D movie premiere tickets (spring blockbuster),Entertainment,CREDIT_CARD,ECOMMERCE
158.90,2025-03-25,Vehicle refueling,Transportation,CASH,OFFLINE_MARKET
1285.60,2025-03-26,Vehicle insurance renewal,Transportation,CREDIT_CARD,ECOMMERCE
125.40,2025-03-26,April rent deposit,Housing,NET_BANKING,UTILITY
195.80,2025-03-27,Motorcycle servicing,Transportation,CASH,OFFLINE_MARKET
315.40,2025-03-27,Prescription medication refill,Health,DEBIT_CARD,OFFLINE_MARKET
255.80,2025-03-28,Quarterly train pass renewal,Transportation,CASH,OFFLINE_MARKET
185.60,2025-03-28,Online course subscription,Entertainment,CREDIT_CARD,ECOMMERCE
455.40,2025-03-29,Business casual attire (spring collection),Shopping,CREDIT_CARD,MALL
98.70,2025-03-29,Professional car cleaning,Transportation,CASH,OFFLINE_MARKET
295.40,2025-03-30,Dinner with old friends,Dining,CASH,RESTAURANT
325.80,2025-03-30,Spring collection accessories,Shopping,CREDIT_CARD,MALL
365.90,2025-03-31,Train ticket to Chennai (holiday travel increase),Transportation,CASH,OFFLINE_MARKET
545.60,2025-03-31,Designer watch purchase,Shopping,CREDIT_CARD,MALL
212.45,2025-04-01,Weekly groceries (price increase),Groceries,UPI,QUICK_COMMERCE
432.90,2025-04-01,Spring festival tickets,Entertainment,CREDIT_CARD,ECOMMERCE
165.30,2025-04-02,Electricity bill April,Utilities,UPI,UTILITY
1750.60,2025-04-03,Summer wardrobe shopping,Shopping,CREDIT_CARD,MALL
65.40,2025-04-03,Daily commute expenses,Transportation,CASH,OFFLINE_MARKET
425.90,2025-04-04,Quarterly health check-up,Health,DEBIT_CARD,OFFLINE_MARKET
95.80,2025-04-04,Mobile data plan renewal,Utilities,UPI,UTILITY
710.30,2025-04-05,Family dinner (special spring menu),Dining,CREDIT_CARD,RESTAURANT
25.40,2025-04-05,Metro transit card top-up,Transportation,CASH,OFFLINE_MARKET
62.80,2025-04-06,Office snacks and refreshments,Groceries,CASH,OFFLINE_MARKET
29500.00,2025-04-06,April apartment rent (seasonal increase),Housing,NET_BANKING,UTILITY
82.70,2025-04-07,Mobile phone bill,Utilities,UPI,UTILITY
295.60,2025-04-07,New fiction books collection,Entertainment,CREDIT_CARD,ECOMMERCE
75.30,2025-04-08,Taxi to business meeting,Transportation,CASH,OFFLINE_MARKET
135.45,2025-04-08,Fresh seasonal produce,Groceries,UPI,QUICK_COMMERCE
392.20,2025-04-09,Seasonal allergy medication,Health,DEBIT_CARD,OFFLINE_MARKET
255.75,2025-04-09,Weekend concert tickets,Entertainment,CREDIT_CARD,ECOMMERCE
495.30,2025-04-10,Team building lunch,Food & Beverage,CASH,RESTAURANT
82.45,2025-04-10,Highway toll payment,Transportation,CASH,OFFLINE_MARKET
145.60,2025-04-11,Premium dairy products,Groceries,UPI,QUICK_COMMERCE
945.75,2025-04-11,Summer accessory shopping,Shopping,CREDIT_CARD,MALL
52.30,2025-04-12,Water bill payment,Utilities,UPI,UTILITY
280.00,2025-04-12,Spring salon makeover,Personal Care,CASH,OFFLINE_MARKET
72.50,2025-04-13,Vehicle maintenance,Transportation,CASH,OFFLINE_MARKET
255.75,2025-04-13,Fresh vegetables and fruits (seasonal),Groceries,UPI,QUICK_COMMERCE
580.30,2025-04-14,Annual eye checkup,Health,DEBIT_CARD,OFFLINE_MARKET
135.90,2025-04-14,Gaming subscription renewal,Entertainment,CREDIT_CARD,ECOMMERCE
515.25,2025-04-15,Easter holiday brunch,Food & Beverage,CASH,RESTAURANT
92.40,2025-04-15,Airport transportation,Transportation,CASH,OFFLINE_MARKET
210.30,2025-04-16,Spring cleaning supplies,Household,CASH,OFFLINE_MARKET
"""
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
                payment_method = row.get('payment_method', 'CASH')  # Default to CASH if not provided
                transaction_category = row.get('transaction_category', 'OTHER')  # Default to OTHER if not provided
                
                expense = Expense(
                    amount=amount,
                    date=date,
                    description=description,
                    category=category,
                    owner=user,
                    is_recurring='NO',
                    spent_by=user.username,
                    payment_method=payment_method,
                    transaction_category=transaction_category
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
