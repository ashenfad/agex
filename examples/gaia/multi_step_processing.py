"""
GAIA-style Multi-Step Data Processing

Problem: You have raw sensor data, calibration constants, and device metadata.
Process the sensor data using the calibration constants, filter based on device
metadata, and generate a summary report with insights.

This demonstrates:
- Working with pre-loaded data objects (DataFrame, dict)
- Data cleaning and transformation
- Cross-referencing between datasets
- Statistical analysis
- Rich object passing (no file access needed)
"""

import numpy as np
import pandas as pd

from agex import Agent
from agex.helpers import register_numpy, register_pandas, register_stdlib

# Create agent with comprehensive data processing capabilities
agent = Agent(
    name="data_processor",
    primer="You are an expert data analyst who can process multiple data formats and generate insights.",
)
register_pandas(agent)
register_numpy(agent)
register_stdlib(agent)


@agent.task
def analyze_sensor_data(
    task_description: str, **kwargs: pd.DataFrame | dict | float | int | str
) -> float | int:  # type: ignore[return-value]
    """
    Analyze sensor data to answer various questions.

    Example tasks:
    - "Calculate average calibrated temperature for temperature sensors on 2023-01-15"
    - "Count sensors with calibration drift above 15% threshold"
    - "Calculate total downtime minutes for Building A temperature sensors"
    - "Calculate energy cost for Building A temperature control"
    - "Calculate average calibration age in days from 2023-01-31"

    Args:
        task_description: Natural language description of the analysis task
        **kwargs: Available data including sensor_df, calibration_df, metadata_dict,
                 and any relevant parameters like thresholds, dates, costs, etc.

    Return:
        Numerical result (float for costs/averages, int for counts/days/minutes)
    """
    pass


def create_sample_datasets():
    """Create sample datasets for testing"""

    # Create raw sensor data (DataFrame)
    np.random.seed(42)  # For reproducible results

    timestamps = pd.date_range("2023-01-01", periods=1000, freq="H")
    sensor_ids = ["TEMP_001", "TEMP_002", "HUMID_001", "HUMID_002", "PRESS_001"]

    sensor_data = []
    for timestamp in timestamps:
        for sensor_id in sensor_ids:
            # Simulate different sensor behaviors
            if "TEMP" in sensor_id:
                base_value = 20 + 10 * np.sin(2 * np.pi * timestamp.hour / 24)
                noise = np.random.normal(0, 2)
            elif "HUMID" in sensor_id:
                base_value = 50 + 20 * np.sin(
                    2 * np.pi * timestamp.hour / 24 + np.pi / 2
                )
                noise = np.random.normal(0, 3)
            else:  # PRESS
                base_value = 1013 + 5 * np.sin(2 * np.pi * timestamp.hour / 24)
                noise = np.random.normal(0, 1)

            # Add some outliers
            if np.random.random() < 0.02:  # 2% outliers
                noise *= 10

            sensor_data.append(
                {
                    "timestamp": timestamp,
                    "sensor_id": sensor_id,
                    "raw_value": base_value + noise,
                }
            )

    sensor_df = pd.DataFrame(sensor_data)

    # Create calibration constants (DataFrame)
    calibration_data = {
        "sensor_id": ["TEMP_001", "TEMP_002", "HUMID_001", "HUMID_002", "PRESS_001"],
        "offset": [-0.5, 0.3, -2.1, 1.8, 0.0],
        "scale_factor": [1.02, 0.98, 1.05, 0.95, 1.001],
        "last_calibrated": [
            "2023-01-01",
            "2023-01-15",
            "2022-12-20",
            "2023-01-10",
            "2023-01-05",
        ],
    }

    calibration_df = pd.DataFrame(calibration_data)

    # Create device metadata (dict)
    metadata = {
        "devices": {
            "TEMP_001": {
                "location": "Building A - Room 101",
                "device_type": "Temperature",
                "manufacturer": "SensorTech",
                "install_date": "2022-06-15",
                "maintenance_schedule": "quarterly",
                "accuracy": 0.1,
                "expected_range": [15, 35],
            },
            "TEMP_002": {
                "location": "Building A - Room 102",
                "device_type": "Temperature",
                "manufacturer": "SensorTech",
                "install_date": "2022-06-15",
                "maintenance_schedule": "quarterly",
                "accuracy": 0.1,
                "expected_range": [15, 35],
            },
            "HUMID_001": {
                "location": "Building A - Room 101",
                "device_type": "Humidity",
                "manufacturer": "WeatherSense",
                "install_date": "2022-07-01",
                "maintenance_schedule": "monthly",
                "accuracy": 2.0,
                "expected_range": [20, 80],
            },
            "HUMID_002": {
                "location": "Building A - Room 102",
                "device_type": "Humidity",
                "manufacturer": "WeatherSense",
                "install_date": "2022-07-01",
                "maintenance_schedule": "monthly",
                "accuracy": 2.0,
                "expected_range": [20, 80],
            },
            "PRESS_001": {
                "location": "Building A - Roof",
                "device_type": "Pressure",
                "manufacturer": "AtmosPro",
                "install_date": "2022-05-20",
                "maintenance_schedule": "semi-annual",
                "accuracy": 0.5,
                "expected_range": [1000, 1030],
            },
        }
    }

    return sensor_df, calibration_df, metadata


def main():
    print("Creating sample datasets...")
    sensor_df, calibration_df, metadata = create_sample_datasets()

    # Show what we're working with
    print("Dataset info:")
    print(f"- Sensor readings: {len(sensor_df)} records")
    print(f"- Unique sensors: {sensor_df['sensor_id'].nunique()}")
    print(
        f"- Date range: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}"
    )
    print(f"- Calibration data: {len(calibration_df)} sensors")
    print(f"- Device metadata: {len(metadata['devices'])} devices")

    print("\n=== Calibrated Temperature Analysis ===")
    # Pass objects directly to the agent - no file access needed!
    avg_temp = analyze_sensor_data(
        "Calculate average calibrated temperature for temperature sensors on 2023-01-15",
        sensor_df=sensor_df,
        calibration_df=calibration_df,
        metadata_dict=metadata,
    )

    print(f"Average calibrated temperature on 2023-01-15: {avg_temp:.2f}°C")

    # print("\n=== Calibration Age Analysis ===")
    # avg_calibration_age = analyze_sensor_data(
    #     "Calculate average calibration age in days from 2023-01-31",
    #     calibration_df=calibration_df,
    # )

    # print(f"Average calibration age: {avg_calibration_age} days")

    # print("\n=== Sensor Downtime Calculation ===")
    # downtime = analyze_sensor_data(
    #     "Calculate total downtime minutes for Building A temperature sensors",
    #     sensor_df=sensor_df,
    #     metadata_dict=metadata,
    # )

    # print(f"Total downtime minutes: {downtime}")

    # print("\n=== Calibration Drift Detection ===")
    # drift_count = analyze_sensor_data(
    #     "Count sensors with calibration drift above 15% threshold",
    #     sensor_df=sensor_df,
    #     calibration_df=calibration_df,
    #     metadata_dict=metadata,
    # )

    # print(f"Sensors with calibration drift: {drift_count}")

    # print("\n=== Building Energy Cost Calculation ===")
    # energy_cost = analyze_sensor_data(
    #     "Calculate energy cost for Building A temperature control",
    #     sensor_df=sensor_df,
    #     metadata_dict=metadata,
    # )

    # print(f"Total energy cost: ${energy_cost:.2f}")

    # # Manual verification of some statistics
    # print(f"\n=== Manual Verification ===")
    # print(f"Total raw records: {len(sensor_df)}")
    # print(f"Unique sensors: {sensor_df['sensor_id'].nunique()}")
    # print(
    #     f"Date range: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}"
    # )


if __name__ == "__main__":
    main()
