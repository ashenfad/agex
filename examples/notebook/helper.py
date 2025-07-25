import sqlite3

import numpy as np
import pandas as pd

from agex.llm.core import LLMResponse


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


# Canned LLM responses for demo reproducibility
# These are actual responses captured from real LLM calls, used here for fast, reliable demos

DB_CODE = """
import pandas as pd
query = "SELECT date, product_name, quantity_sold, price_per_unit FROM sales WHERE product_name = 'Umbrellas' ORDER BY date"
with db.execute(query) as cursor:
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
task_success(df)
"""

DB_QUERY_RESPONSE = LLMResponse(
    thinking="I need to execute the SQL query and convert the results to a pandas DataFrame.",
    code=DB_CODE,
)

VIZ_CODE = """
import pandas as pd
import plotly.express as px
# Extract year and day-of-year for overlaying cycles
inputs.df['date'] = pd.to_datetime(inputs.df['date'])
inputs.df['year'] = inputs.df['date'].dt.year
inputs.df['day_of_year'] = inputs.df['date'].dt.dayofyear

fig = px.line(inputs.df, x='day_of_year', y='quantity_sold', color='year', 
              title="Yearly Sales Cycles Overlayed",
              labels={'day_of_year': 'Day of Year', 'quantity_sold': 'Quantity Sold'})
task_success(fig)
"""

VISUALIZATION_RESPONSE = LLMResponse(
    thinking="I need to create a line plot showing the quantity sold over time.",
    code=VIZ_CODE,
)
