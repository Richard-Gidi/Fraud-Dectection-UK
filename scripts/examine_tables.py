"""Examine Snowflake table structures."""
import snowflake.connector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Create a cursor
cur = conn.cursor()

# List all tables
print("Available tables:")
cur.execute("SHOW TABLES")
tables = cur.fetchall()
for t in tables:
    print(f"\nTable: {t[1]}")
    cur.execute(f"DESC TABLE {t[1]}")
    columns = cur.fetchall()
    print("Columns:")
    for col in columns:
        print(f"  - {col[0]}: {col[1]}")

cur.close()
conn.close()