import os

# Flask config
SECRET_KEY = 'your-secret-key-change-this-in-production'
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

# Database config
DB_USER = 'postgres'
DB_PASSWORD = 'your_password_here'  # Replace with your actual password
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'aircraft_maintenance'
DATABASE_URI = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Grafana config
GRAFANA_URL = 'http://localhost:3000'