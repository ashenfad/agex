import sqlite3

import numpy as np
import pandas as pd


def create_in_memory_db():
    """
    Creates and populates an in-memory SQLite database with pretend sales data.
    The data includes seasonal trends for different products.
    Returns the sqlite3.Connection object.
    """
    conn = sqlite3.connect(":memory:")

    # Create a DataFrame with sample data
    num_days = 365 * 2  # Two years of data
    start_date = pd.to_datetime("2022-01-01")
    dates = pd.date_range(start_date, periods=num_days, freq="D")

    products = ["Umbrellas", "Sunscreen", "Jackets", "Shorts"]
    data = []

    for date in dates:
        for product in products:
            # Base sales
            base_sales = np.random.randint(10, 50)

            # Seasonal trends
            month = date.month
            if product == "Umbrellas":  # Sells more in rainy seasons (spring/fall)
                seasonality = np.sin(2 * np.pi * (month - 3) / 12) + 1
            elif product == "Sunscreen":  # Sells more in summer
                seasonality = np.sin(2 * np.pi * (month - 6) / 12) + 1
            elif product == "Jackets":  # Sells more in winter
                seasonality = -np.sin(2 * np.pi * (month - 6) / 12) + 1
            elif product == "Shorts":  # Sells more in summer
                seasonality = np.sin(2 * np.pi * (month - 6) / 12) + 1
            else:
                seasonality = 1

            # Random noise
            noise = np.random.normal(0, 5)

            quantity_sold = max(0, int(base_sales * seasonality + noise))
            price = {"Umbrellas": 15, "Sunscreen": 12, "Jackets": 80, "Shorts": 25}[
                product
            ]

            data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "product_name": product,
                    "quantity_sold": quantity_sold,
                    "price_per_unit": price,
                }
            )

    sales_df = pd.DataFrame(data)

    # Write the DataFrame to the SQLite database
    sales_df.to_sql("sales", conn, if_exists="replace", index=False)

    return conn


SETUP_ACTION = """
# find columns in the sales table
columns_info = db.execute("PRAGMA table_info(sales)").fetchall()
columns = [col[1] for col in columns_info]

# find distinct product names
distinct_products = db.execute("SELECT DISTINCT product_name FROM sales").fetchall()
product_names = [row[0] for row in distinct_products]

task_continue("Columns in 'sales' table:", columns, "Distinct product_names:", product_names)
"""
