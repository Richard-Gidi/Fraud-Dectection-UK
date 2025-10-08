"""Load data from Snowflake and save locally."""
import snowflake.connector
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA'),
    role=os.getenv('SNOWFLAKE_ROLE')
)

# Load tables using pandas
def read_table(table_name, conn):
    query = f"SELECT * FROM {table_name};"
    return pd.read_sql(query, conn)

# Read tables from Snowflake
tables = {
    'customer_info': read_table("CUSTOMER_INFO", conn),
    'fraudulent_patterns': read_table("FRAUDULENT_PATTERNS", conn),
    'merchant_info': read_table("MERCHANT_INFO", conn),
    'transactions': read_table("TRANSACTIONS", conn),
    'transactions_flags': read_table("TRANSACTIONS_FLAGS", conn),
    'transaction_patterns': read_table("TRANSACTION_PATTERNS", conn)
}

# Create training data with feature engineering
transactions = tables['transactions'].copy()

# Convert date to datetime and extract features
transactions['TRANSACTION_DATE'] = pd.to_datetime(transactions['TRANSACTION_DATE'])
transactions['hour'] = transactions['TRANSACTION_DATE'].dt.hour
transactions['day_of_week'] = transactions['TRANSACTION_DATE'].dt.dayofweek
transactions['month'] = transactions['TRANSACTION_DATE'].dt.month

# Create numeric features
numeric_features = ['AMOUNT', 'hour', 'day_of_week', 'month']

# Create categorical features using label encoding
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'CHANNEL', 'TRANSACTION_DEVICE']
encoders = {}

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    transactions[f'{col}_encoded'] = encoders[col].fit_transform(transactions[col])

# Select features for training
feature_cols = numeric_features + [f'{col}_encoded' for col in categorical_cols]

# Prepare final training data
training_data = transactions[feature_cols + ['FRAUD_FLAG']].copy()

# Rename FRAUD_FLAG to label for consistency with model training
training_data = training_data.rename(columns={'FRAUD_FLAG': 'label'})

# Fill missing flags as legitimate transactions (0)
training_data['label'] = training_data['label'].fillna(0)

# Save encoders for future use
import joblib
os.makedirs('./models', exist_ok=True)
joblib.dump(encoders, './models/label_encoders.joblib')

# Save to CSV
training_data.to_csv('./data/transactions.csv', index=False)

# Create models directory
os.makedirs('./models', exist_ok=True)

print(f"Saved {len(training_data)} transactions to ./data/transactions.csv")
print(f"Created models directory at ./models")