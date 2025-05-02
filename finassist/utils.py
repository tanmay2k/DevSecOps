import json
from datetime import date, datetime
from decimal import Decimal

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)  # You could also use str(obj) for more precision
        return super().default(obj)

