"""
GAIA-style Excel Analysis Problem

Problem: You have sales data and need to calculate the total revenue
from all "Electronics" category items sold in Q2 2023, excluding any items
with a discount greater than 20%.

This demonstrates:
- Working with pre-loaded DataFrames (no file access needed)
- Data filtering and aggregation
- Multi-condition logic
- Financial calculations
- Rich object passing between host and agent
"""

import pandas as pd

from agex import Agent
from agex.helpers import register_pandas

# Create agent with data processing capabilities
agent = Agent(name="excel_analyst", primer="You are expert at analyzing business data.")
register_pandas(agent)


@agent.task
def analyze_sales_data(sales_df: pd.DataFrame, question: str) -> float:  # type: ignore[return-value]
    """Analyze sales data and answer business questions."""
    pass


def create_sample_data() -> pd.DataFrame:
    """Create sample sales data for testing"""
    import datetime
    import random

    # Generate sample sales data
    categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]

    data = []
    for i in range(500):
        # Random date in 2023 - use datetime.datetime instead of datetime.date
        date = datetime.datetime(2023, random.randint(1, 12), random.randint(1, 28))

        data.append(
            {
                "Order_ID": f"ORD-{i+1:04d}",
                "Date": date,
                "Category": random.choice(categories),
                "Product_Name": f"Product_{i+1}",
                "Unit_Price": round(random.uniform(10, 500), 2),
                "Quantity": random.randint(1, 10),
                "Discount_Percent": round(random.uniform(0, 35), 1),
                "Revenue": 0,  # Will calculate
            }
        )

    # Calculate revenue after discount
    for row in data:
        gross_revenue = row["Unit_Price"] * row["Quantity"]
        discount_amount = gross_revenue * (row["Discount_Percent"] / 100)
        row["Revenue"] = round(gross_revenue - discount_amount, 2)

    df = pd.DataFrame(data)

    # Ensure Date column is properly recognized as datetime
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def main():
    # Create sample data (host loads the data, not the agent)
    print("Creating sample sales data...")
    df = create_sample_data()

    # Show what we're working with
    print("Dataset info:")
    print(f"- Total records: {len(df)}")
    print(f"- Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"- Categories: {df['Category'].unique()}")
    print(f"- Revenue range: ${df['Revenue'].min():.2f} to ${df['Revenue'].max():.2f}")

    # Test the agent with pre-loaded data
    question = """
    Calculate the total revenue from all "Electronics" category items 
    sold in Q2 2023 (April, May, June), excluding any items with a 
    discount greater than 20%. Return the result as a float.
    """

    print("\n=== Agent Analysis ===")
    result = analyze_sales_data(df, question)  # Pass DataFrame directly!
    print(f"Agent result: ${result:,.2f}")

    # Manual verification for comparison
    q2_months = [4, 5, 6]
    filtered_df = df[
        (df["Category"] == "Electronics")
        & (df["Date"].dt.month.isin(q2_months))
        & (df["Discount_Percent"] <= 20)
    ]
    expected = filtered_df["Revenue"].sum()
    print(f"Expected result: ${expected:,.2f}")
    print(f"Match: {'✓' if abs(result - expected) < 0.01 else '✗'}")


if __name__ == "__main__":
    main()
