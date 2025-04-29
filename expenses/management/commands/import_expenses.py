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
455.30,2025-01-23,Multiplex premium screening tickets,Entertainment
135.60,2025-01-23,Car refueling,Transportation
1250.80,2025-01-24,Car insurance quarterly payment,Transportation
115.25,2025-01-24,February rent advance,Housing
185.40,2025-01-25,Motorcycle servicing,Transportation
295.80,2025-01-25,Prescription medications,Health
245.75,2025-01-26,Monthly train pass,Transportation
175.30,2025-01-26,Digital magazine subscription,Entertainment
350.60,2025-01-27,New formal pants for office,Shopping
95.40,2025-01-27,Car wash service,Transportation
185.30,2025-01-28,Dinner with friends,Dining
215.75,2025-01-28,New shirt purchase,Shopping
265.40,2025-01-29,Train ticket to Hyderabad,Transportation
435.90,2025-01-29,Designer sunglasses,Shopping
135.60,2025-01-30,Casual dining experience,Dining
295.40,2025-01-30,Monthly grocery stock-up,Groceries
125.30,2025-01-31,Coffee shop work session,Food & Beverage
42.90,2025-01-31,Scarf purchase,Shopping
335.60,2025-02-01,Internet bill for February,Utilities
115.75,2025-02-01,Comedy show tickets,Entertainment
1580.00,2025-02-02,Traditional outfit purchase,Shopping
520.80,2025-02-02,Anniversary dinner at fine dining,Dining
515.30,2025-02-03,Airport parking for weekend trip,Transportation
1250.75,2025-02-03,Flight tickets for weekend getaway,Transportation
185.40,2025-02-04,Mobile data plan renewal,Utilities
72.50,2025-02-04,Evening cafe visit,Food & Beverage
385.90,2025-02-05,Car service and oil change,Transportation
265.40,2025-02-05,Monthly gym membership renewal,Fitness
88.75,2025-02-06,Scooter maintenance,Transportation
28500.00,2025-02-06,February apartment rent,Housing
38.90,2025-02-07,Comic book purchase,Entertainment
495.80,2025-02-07,Business dinner meeting,Dining
165.30,2025-02-08,Food delivery order,Dining
372.45,2025-02-08,Society maintenance payment,Housing
195.60,2025-02-09,Train ticket to Pune,Transportation
295.30,2025-02-09,Airport shuttle service,Transportation
315.80,2025-02-10,Electricity bill payment,Utilities
125.40,2025-02-10,Lunch with client,Dining
525.90,2025-02-11,Annual sports club membership,Fitness
420.75,2025-02-11,March rent advance payment,Housing
505.30,2025-02-12,Collector's edition book set,Entertainment
485.75,2025-02-12,Business attire purchase,Shopping
345.80,2025-02-13,Family dinner outing,Dining
495.40,2025-02-13,Weekly grocery shopping,Groceries
68.90,2025-02-14,Shopping mall parking,Transportation
1425.60,2025-02-14,Valentine's Day special dinner,Dining
1245.30,2025-02-14,Valentine's Day gift package,Shopping
195.80,2025-02-15,Pilates monthly subscription,Fitness
235.60,2025-02-16,General physician consultation,Health
165.40,2025-02-16,Medical test fees,Health
450.75,2025-02-17,Weekend family brunch,Dining
215.80,2025-02-17,March month rental deposit,Housing
175.40,2025-02-18,Spinning class package,Fitness
305.25,2025-02-18,Home broadband bill payment,Utilities
98.75,2025-02-19,History book purchase,Entertainment
440.80,2025-02-19,Petrol fill-up,Transportation
65.30,2025-02-20,Accessories shopping,Shopping
545.90,2025-02-20,Business networking dinner,Dining
425.75,2025-02-21,Business lunch meeting,Dining
135.60,2025-02-21,Health supplements,Health
22.50,2025-02-22,Afternoon tea break,Food & Beverage
445.90,2025-02-22,Bus tickets for family trip,Transportation
395.80,2025-02-23,Scooter fuel fill-up,Transportation
405.60,2025-02-23,Weekly grocery shopping,Groceries
420.75,2025-02-24,Electric bill payment,Utilities
65.40,2025-02-24,Takeout dinner,Dining
215.80,2025-02-25,Movie rental subscription,Entertainment
85.60,2025-02-25,Vehicle cleaning service,Transportation
475.90,2025-02-26,Special occasion dinner,Dining
28500.00,2025-02-26,March apartment rent,Housing
195.80,2025-02-27,Fresh groceries shopping,Groceries
58.40,2025-02-27,Street food festival visit,Food & Beverage
185.90,2025-02-28,Leather handbag purchase,Shopping
195.40,2025-02-28,Skin specialist consultation,Health
395.80,2025-03-01,Airport parking fee,Transportation
355.60,2025-03-01,April month rent advance,Housing
485.90,2025-03-02,Family Sunday brunch,Dining
275.60,2025-03-02,Dance class monthly fee,Fitness
185.30,2025-03-03,Train ticket to Kolkata,Transportation
48.90,2025-03-03,Morning coffee and pastry,Food & Beverage
265.40,2025-03-04,Grocery shopping at D-Mart,Groceries
142.80,2025-03-04,March electricity bill (increased due to spring),Utilities
1680.50,2025-03-05,Spring collection dress,Shopping
58.90,2025-03-05,Daily office parking,Transportation
415.75,2025-03-06,Annual medical insurance co-pay,Health
88.60,2025-03-06,Mobile internet plan,Utilities
645.30,2025-03-07,Anniversary dinner celebration,Dining
22.50,2025-03-07,Local bus day pass,Transportation
55.80,2025-03-08,Office break snacks,Groceries
29000.00,2025-03-08,March month apartment rent,Housing
75.40,2025-03-09,Mobile phone monthly bill,Utilities
235.60,2025-03-09,Biography book purchase,Entertainment
68.90,2025-03-10,Shared auto ride,Transportation
125.40,2025-03-10,Vegetables and fruits,Groceries
355.80,2025-03-11,Pharmacy purchase,Health
205.40,2025-03-11,Concert tickets,Entertainment
525.90,2025-03-12,Office team lunch (increased for special event),Food & Beverage
75.60,2025-03-12,Highway toll charges,Transportation
135.80,2025-03-13,Weekly bread and dairy,Groceries
1265.30,2025-03-13,Spring wardrobe update (seasonal increase),Shopping
45.60,2025-03-14,Water utility bill,Utilities
295.40,2025-03-14,Spa treatment,Personal Care
65.30,2025-03-15,Electric scooter maintenance,Transportation
315.40,2025-03-15,Seasonal fruits shopping (spring produce increase),Groceries
535.80,2025-03-16,ENT specialist consultation,Health
115.40,2025-03-16,Music subscription annual plan,Entertainment
525.90,2025-03-17,St. Patrick's Day celebration,Food & Beverage
85.60,2025-03-17,Rideshare to airport,Transportation
165.40,2025-03-18,Home cleaning supplies,Household
895.80,2025-03-18,Spring cleaning service,Household
35.60,2025-03-19,Local train tickets,Transportation
525.40,2025-03-19,Dental check-up and cleaning,Health
405.90,2025-03-20,Monthly parking facility fee,Transportation
465.40,2025-03-20,Car fuel fill-up (seasonal travel increase),Transportation
215.80,2025-03-21,Fitness class package (spring fitness resolution),Fitness
118.90,2025-03-21,Diesel purchase for generator,Transportation
52.40,2025-03-22,Metro card top-up,Transportation
485.60,2025-03-22,Home internet monthly bill,Utilities
455.40,2025-03-23,Weekly grocery shopping (increased for spring parties),Groceries
515.80,2025-03-23,Dinner at new fusion restaurant,Dining
32.50,2025-03-24,Prepaid mobile recharge,Utilities
635.60,2025-03-24,Theater festival tickets (spring cultural events),Entertainment
575.40,2025-03-25,3D movie premiere tickets (spring blockbuster),Entertainment
158.90,2025-03-25,Vehicle refueling,Transportation
1285.60,2025-03-26,Vehicle insurance renewal,Transportation
125.40,2025-03-26,April rent deposit,Housing
195.80,2025-03-27,Motorcycle servicing,Transportation
315.40,2025-03-27,Prescription medication refill,Health
255.80,2025-03-28,Quarterly train pass renewal,Transportation
185.60,2025-03-28,Online course subscription,Entertainment
455.40,2025-03-29,Business casual attire (spring collection),Shopping
98.70,2025-03-29,Professional car cleaning,Transportation
295.40,2025-03-30,Dinner with old friends,Dining
325.80,2025-03-30,Spring collection accessories,Shopping
365.90,2025-03-31,Train ticket to Chennai (holiday travel increase),Transportation
545.60,2025-03-31,Designer watch purchase,Shopping
212.45,2025-04-01,Weekly groceries (price increase),Groceries
432.90,2025-04-01,Spring festival tickets,Entertainment
165.30,2025-04-02,Electricity bill April,Utilities
1750.60,2025-04-03,Summer wardrobe shopping,Shopping
65.40,2025-04-03,Daily commute expenses,Transportation
425.90,2025-04-04,Quarterly health check-up,Health
95.80,2025-04-04,Mobile data plan renewal,Utilities
710.30,2025-04-05,Family dinner (special spring menu),Dining
25.40,2025-04-05,Metro transit card top-up,Transportation
62.80,2025-04-06,Office snacks and refreshments,Groceries
29500.00,2025-04-06,April apartment rent (seasonal increase),Housing
82.70,2025-04-07,Mobile phone bill,Utilities
295.60,2025-04-07,New fiction books collection,Entertainment
75.30,2025-04-08,Taxi to business meeting,Transportation
135.45,2025-04-08,Fresh seasonal produce,Groceries
392.20,2025-04-09,Seasonal allergy medication,Health
255.75,2025-04-09,Weekend concert tickets,Entertainment
495.30,2025-04-10,Team building lunch,Food & Beverage
82.45,2025-04-10,Highway toll payment,Transportation
145.60,2025-04-11,Premium dairy products,Groceries
945.75,2025-04-11,Summer accessory shopping,Shopping
52.30,2025-04-12,Water bill payment,Utilities
280.00,2025-04-12,Spring salon makeover,Personal Care
72.50,2025-04-13,Vehicle maintenance,Transportation
255.75,2025-04-13,Fresh vegetables and fruits (seasonal),Groceries
580.30,2025-04-14,Annual eye checkup,Health
135.90,2025-04-14,Gaming subscription renewal,Entertainment
515.25,2025-04-15,Easter holiday brunch,Food & Beverage
92.40,2025-04-15,Airport transportation,Transportation
210.30,2025-04-16,Spring cleaning supplies,Household
890.50,2025-04-16,Patio furniture (seasonal),Household
42.45,2025-04-17,Local transportation,Transportation
595.90,2025-04-17,Dermatologist consultation,Health
425.80,2025-04-18,Monthly parking pass renewal,Transportation
375.60,2025-04-18,Fuel for extended weekend travel,Transportation
155.40,2025-04-19,Yoga retreat (spring special),Fitness
132.30,2025-04-19,Generator maintenance,Transportation
60.75,2025-04-20,Metro card recharge,Transportation
495.90,2025-04-20,Internet bill payment,Utilities
380.80,2025-04-21,Grocery shopping for Easter weekend,Groceries
458.50,2025-04-21,Dinner at Italian restaurant,Dining
38.45,2025-04-22,Mobile recharge,Utilities
625.75,2025-04-22,Spring festival tickets,Entertainment
555.30,2025-04-23,Outdoor movie event tickets,Entertainment
155.60,2025-04-23,Car refueling,Transportation
215.40,2025-04-24,Vehicle service package,Transportation
145.25,2025-04-24,May rent advance,Housing
215.40,2025-04-25,Bicycle seasonal maintenance,Transportation
345.80,2025-04-25,Seasonal health supplements,Health
285.75,2025-04-26,Monthly transit pass,Transportation
195.30,2025-04-26,Digital subscription renewal,Entertainment
450.60,2025-04-27,Summer casual wear,Shopping
115.40,2025-04-27,Vehicle detailing service,Transportation
215.30,2025-04-28,Dinner with colleagues,Dining
255.75,2025-04-28,Summer hat and accessories,Shopping
295.40,2025-04-29,Train ticket for holiday weekend,Transportation
485.90,2025-04-29,Designer summer clothing,Shopping
175.60,2025-04-30,Outdoor dining experience,Dining
345.40,2025-04-30,Monthly grocery stock-up,Groceries"""
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
