"""
FactoryGuard AI - IoT Sensor Data Generator
Generates realistic sensor data for 500 robotic arms with failure patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class IoTSensorSimulator:
    def __init__(self, n_robots=500, days=90, seed=42):
        """
        Generate synthetic IoT sensor data for predictive maintenance

        Args:
            n_robots: Number of robotic arms to simulate
            days: Number of days of historical data
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.n_robots = n_robots
        self.days = days
        self.failure_rate = 0.01  # 1% failure rate (class imbalance)

    def generate_normal_operation(self, hours):
        """Generate sensor readings for normal operation"""
        vibration = np.random.normal(0.3, 0.05, hours)  # Normal: 0.3 ± 0.05
        temperature = np.random.normal(65, 5, hours)  # Normal: 65°C ± 5
        pressure = np.random.normal(150, 10, hours)  # Normal: 150 PSI ± 10

        # Add realistic noise and daily patterns
        time_factor = np.sin(np.linspace(0, 2 * np.pi * self.days, hours)) * 0.02
        vibration += time_factor
        temperature += time_factor * 3

        return vibration, temperature, pressure

    def generate_degradation_pattern(self, hours_before_failure=24):
        """Generate sensor readings showing degradation before failure"""
        hours = hours_before_failure

        # Progressive degradation over 24 hours
        degradation = np.linspace(0, 1, hours)

        # Vibration increases significantly
        vibration = 0.3 + degradation * 0.4 + np.random.normal(0, 0.03, hours)

        # Temperature rises
        temperature = 65 + degradation * 25 + np.random.normal(0, 3, hours)

        # Pressure becomes erratic
        pressure = 150 + degradation * 30 * np.sin(np.linspace(0, 4 * np.pi, hours))
        pressure += np.random.normal(0, 5, hours)

        return vibration, temperature, pressure

    def generate_dataset(self):
        """Generate complete dataset with failures"""
        data = []
        start_date = datetime.now() - timedelta(days=self.days)

        for robot_id in range(self.n_robots):
            hours_total = self.days * 24
            current_hour = 0

            while current_hour < hours_total:
                # Decide if this robot will fail
                if np.random.random() < self.failure_rate:
                    # Generate normal operation until 24 hours before failure
                    hours_normal = np.random.randint(48, 200)
                    if current_hour + hours_normal + 24 > hours_total:
                        hours_normal = hours_total - current_hour
                        vib, temp, press = self.generate_normal_operation(hours_normal)
                        failure = np.zeros(hours_normal)
                    else:
                        vib, temp, press = self.generate_normal_operation(hours_normal)
                        failure = np.zeros(hours_normal)

                        # Add degradation pattern (24 hours before failure)
                        vib_deg, temp_deg, press_deg = self.generate_degradation_pattern(24)
                        vib = np.concatenate([vib, vib_deg])
                        temp = np.concatenate([temp, temp_deg])
                        press = np.concatenate([press, press_deg])

                        # Label last 24 hours as "will fail"
                        failure = np.concatenate([failure, np.ones(24)])
                        hours_normal += 24

                    current_hour += hours_normal
                else:
                    # Generate normal operation
                    hours_remaining = hours_total - current_hour
                    vib, temp, press = self.generate_normal_operation(hours_remaining)
                    failure = np.zeros(hours_remaining)
                    current_hour = hours_total

                # Create records
                for i in range(len(vib)):
                    timestamp = start_date + timedelta(hours=current_hour - len(vib) + i)
                    data.append({
                        'timestamp': timestamp,
                        'robot_id': f'ARM_{robot_id:03d}',
                        'vibration': vib[i],
                        'temperature': temp[i],
                        'pressure': press[i],
                        'failure_within_24h': int(failure[i])
                    })

        df = pd.DataFrame(data)
        df = df.sort_values(['robot_id', 'timestamp']).reset_index(drop=True)

        # Add some missing values (realistic IoT scenario)
        missing_mask = np.random.random(len(df)) < 0.02  # 2% missing
        df.loc[missing_mask, 'vibration'] = np.nan

        return df


# Generate dataset
print("Generating IoT sensor data for 500 robotic arms...")
print("This simulates 90 days of hourly sensor readings")
print("-" * 60)

simulator = IoTSensorSimulator(n_robots=500, days=90, seed=42)
df = simulator.generate_dataset()

# Save dataset
df.to_csv('sensor_data_raw.csv', index=False)

print(f"\n✓ Dataset generated: {len(df):,} records")
print(f"✓ Robots: {df['robot_id'].nunique()}")
print(f"✓ Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"✓ Failure rate: {df['failure_within_24h'].mean():.2%}")
print(f"✓ Missing values: {df['vibration'].isna().sum()} ({df['vibration'].isna().mean():.2%})")
print(f"\n✓ Saved to: sensor_data_raw.csv")

# Display sample
print("\n" + "=" * 60)
print("SAMPLE DATA (Normal Operation):")
print("=" * 60)
print(df[df['failure_within_24h'] == 0].head(10))

print("\n" + "=" * 60)
print("SAMPLE DATA (Pre-Failure Pattern):")
print("=" * 60)
print(df[df['failure_within_24h'] == 1].head(10))

