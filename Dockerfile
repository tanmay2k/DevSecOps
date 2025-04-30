# 1) Base image
FROM python:3.10-slim

# 2) Prevent Python from writing .pyc files and buffer stalls
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3) Create app dir
WORKDIR /app

# 4) Install system‑level deps for psycopg2 and build tools
RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential \
    pkg-config \
  && rm -rf /var/lib/apt/lists/*

# 5) Copy & install Python requirements
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 6) Pre‑download NLTK data (stopwords and punkt)
RUN python - <<EOF
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
EOF

# 7) Copy the rest of your Django project
COPY . /app/

# 8) Expose the port
EXPOSE 8000

# 9) On container start: make migrations, migrate, then runserver
CMD ["sh", "-c", "\
    python manage.py makemigrations finassist expenses userincome userpreferences userprofile goals&& \
    python manage.py migrate && \
    python manage.py runserver 0.0.0.0:8000\
"]

