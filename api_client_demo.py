"""
FactoryGuard AI - API Client Demo
Demonstrates real-world usage of the prediction API
"""

import requests
import json
import time
from datetime import datetime, timedelta
import random


class FactoryGuardClient:
    """
    Production client for FactoryGuard AI API
    """

    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self):
        """Check if API is healthy"""
        try:
            response = self.session.get(f'{self.base_url}/health', timeout=5)
            return response.json()
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def predict(self, robot_id, sensor_readings, timestamp=None):
        """
        Get failure prediction for a robot

        Args:
            robot_id: Robot identifier (e.g., 'ARM_247')
            sensor_readings: Dict with vibration, temperature, pressure arrays
            timestamp: ISO timestamp (defaults to now)

        Returns:
            Prediction response dict
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        payload = {
            'robot_id': robot_id,
            'sensor_readings': sensor_readings,
            'timestamp': timestamp
        }

        try:
            response = self.session.post(
                f'{self.base_url}/predict',
                json=payload,
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'error': str(e)}

    def batch_predict(self, robots_data):
        """
        Batch prediction for multiple robots

        Args:
            robots_data: List of robot data dicts

        Returns:
            Batch prediction response
        """
        payload = {'robots': robots_data}

        try:
            response = self.session.post(
                f'{self.base_url}/batch_predict',
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'error': str(e)}

    def get_model_info(self):
        """Get information about deployed model"""
        try:
            response = self.session.get(f'{self.base_url}/model/info', timeout=5)
            return response.json()
        except Exception as e:
            return {'error': str(e)}


def generate_sensor_data(hours=12, failure_mode=False):
    """Generate realistic sensor data for testing"""
    if failure_mode:
        # Degradation pattern
        base_vibration = 0.3
        base_temp = 65
        base_pressure = 150

        degradation = [i / hours for i in range(hours)]

        vibration = [base_vibration + d * 0.4 + random.uniform(-0.03, 0.03)
                     for d in degradation]
        temperature = [base_temp + d * 25 + random.uniform(-3, 3)
                       for d in degradation]
        pressure = [base_pressure + d * 30 * (1 if i % 2 == 0 else -1) + random.uniform(-5, 5)
                    for i, d in enumerate(degradation)]
    else:
        # Normal operation
        vibration = [random.uniform(0.25, 0.35) for _ in range(hours)]
        temperature = [random.uniform(60, 70) for _ in range(hours)]
        pressure = [random.uniform(140, 160) for _ in range(hours)]

    return {
        'vibration': vibration,
        'temperature': temperature,
        'pressure': pressure
    }


def demo_single_prediction(client):
    """Demo: Single robot prediction"""
    print("\n" + "=" * 60)
    print("DEMO 1: Single Robot Prediction")
    print("=" * 60)

    # Normal operation
    print("\n1. Testing normal operation...")
    normal_data = generate_sensor_data(hours=12, failure_mode=False)

    result = client.predict(
        robot_id='ARM_001',
        sensor_readings=normal_data
    )

    if 'error' not in result:
        print(f"✓ Robot: {result['robot_id']}")
        print(f"✓ Risk Level: {result['risk_level']}")
        print(f"✓ Failure Probability: {result['failure_probability']:.2%}")
        print(f"✓ Response Time: {result['response_time_ms']:.2f}ms")
        print(f"\nTop Contributing Factors:")
        for i, factor in enumerate(result['top_contributing_factors'][:3], 1):
            print(f"  {i}. {factor['feature']}: {factor['importance']:.4f}")
    else:
        print(f"❌ Error: {result['error']}")

    # Failure pattern
    print("\n2. Testing degradation pattern...")
    failure_data = generate_sensor_data(hours=12, failure_mode=True)

    result = client.predict(
        robot_id='ARM_247',
        sensor_readings=failure_data
    )

    if 'error' not in result:
        print(f"✓ Robot: {result['robot_id']}")
        print(f"✓ Risk Level: {result['risk_level']}")
        print(f"✓ Failure Probability: {result['failure_probability']:.2%}")
        print(f"✓ Predicted Failure Time: {result.get('predicted_failure_time', 'N/A')}")
        print(f"✓ Response Time: {result['response_time_ms']:.2f}ms")
        print(f"\nTop Contributing Factors:")
        for i, factor in enumerate(result['top_contributing_factors'][:3], 1):
            print(f"  {i}. {factor['feature']}: {factor['importance']:.4f}")
    else:
        print(f"❌ Error: {result['error']}")


def demo_batch_prediction(client):
    """Demo: Batch prediction for multiple robots"""
    print("\n" + "=" * 60)
    print("DEMO 2: Batch Prediction (10 Robots)")
    print("=" * 60)

    robots_data = []
    for i in range(10):
        failure_mode = i % 3 == 0  # Every 3rd robot has degradation

        robots_data.append({
            'robot_id': f'ARM_{i:03d}',
            'sensor_readings': generate_sensor_data(hours=12, failure_mode=failure_mode),
            'timestamp': datetime.now().isoformat()
        })

    result = client.batch_predict(robots_data)

    if 'error' not in result:
        print(f"\n✓ Processed: {result['count']} predictions")
        print(f"✓ Total Time: {result['total_time_ms']:.2f}ms")
        print(f"✓ Avg Time per Prediction: {result['avg_time_per_prediction_ms']:.2f}ms")

        print("\nResults Summary:")
        high_risk = sum(1 for p in result['predictions'] if p['risk_level'] == 'HIGH')
        medium_risk = sum(1 for p in result['predictions'] if p['risk_level'] == 'MEDIUM')
        low_risk = sum(1 for p in result['predictions'] if p['risk_level'] == 'LOW')

        print(f"  HIGH Risk:   {high_risk} robots")
        print(f"  MEDIUM Risk: {medium_risk} robots")
        print(f"  LOW Risk:    {low_risk} robots")

        print("\nHigh Risk Robots:")
        for pred in result['predictions']:
            if pred['risk_level'] == 'HIGH':
                print(f"  - {pred['robot_id']}: {pred['failure_probability']:.2%} probability")
    else:
        print(f"❌ Error: {result['error']}")


def demo_performance_test(client, num_requests=100):
    """Demo: Performance testing"""
    print("\n" + "=" * 60)
    print(f"DEMO 3: Performance Test ({num_requests} requests)")
    print("=" * 60)

    latencies = []

    print(f"\nSending {num_requests} prediction requests...")
    start_time = time.time()

    for i in range(num_requests):
        sensor_data = generate_sensor_data(hours=12, failure_mode=random.random() < 0.1)

        result = client.predict(
            robot_id=f'ARM_{i % 500:03d}',
            sensor_readings=sensor_data
        )

        if 'error' not in result:
            latencies.append(result['response_time_ms'])

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_requests}")

    total_time = time.time() - start_time

    if latencies:
        print(f"\n✓ Performance Results:")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {num_requests / total_time:.2f} req/sec")
        print(f"  p50 Latency: {sorted(latencies)[len(latencies) // 2]:.2f}ms")
        print(f"  p95 Latency: {sorted(latencies)[int(len(latencies) * 0.95)]:.2f}ms")
        print(f"  p99 Latency: {sorted(latencies)[int(len(latencies) * 0.99)]:.2f}ms")
        print(f"  Max Latency: {max(latencies):.2f}ms")

        if sorted(latencies)[int(len(latencies) * 0.95)] < 50:
            print("\n✓ SLA MET: p95 latency < 50ms")
        else:
            print("\n⚠️  SLA MISSED: p95 latency >= 50ms")
    else:
        print("❌ No successful requests")


def demo_monitoring(client):
    """Demo: Real-time monitoring simulation"""
    print("\n" + "=" * 60)
    print("DEMO 4: Real-Time Monitoring (30 seconds)")
    print("=" * 60)

    print("\nSimulating continuous monitoring of 10 robots...")
    print("Press Ctrl+C to stop\n")

    robot_ids = [f'ARM_{i:03d}' for i in range(10)]

    try:
        start_time = time.time()
        iteration = 0

        while time.time() - start_time < 30:
            iteration += 1

            # Pick random robot
            robot_id = random.choice(robot_ids)

            # Generate sensor data (10% chance of degradation)
            sensor_data = generate_sensor_data(
                hours=12,
                failure_mode=random.random() < 0.1
            )

            result = client.predict(
                robot_id=robot_id,
                sensor_readings=sensor_data
            )

            if 'error' not in result:
                risk_emoji = {
                    'HIGH': '🔴',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }

                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{risk_emoji[result['risk_level']]} "
                      f"{result['robot_id']} - "
                      f"Risk: {result['risk_level']} "
                      f"({result['failure_probability']:.1%}) - "
                      f"{result['response_time_ms']:.1f}ms")

            time.sleep(1)

        print(f"\n✓ Monitored {iteration} predictions over 30 seconds")

    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")


# Main demo execution
if __name__ == '__main__':
    print("=" * 60)
    print("FACTORYGUARD AI - API CLIENT DEMO")
    print("=" * 60)

    # Initialize client
    client = FactoryGuardClient(base_url='http://localhost:5000')

    # Health check
    print("\nChecking API health...")
    health = client.health_check()
    if health.get('status') == 'healthy':
        print("✓ API is healthy and ready")
    else:
        print(f"❌ API health check failed: {health}")
        exit(1)

    # Get model info
    print("\nRetrieving model information...")
    model_info = client.get_model_info()
    if 'error' not in model_info:
        print(f"✓ Model Type: {model_info['model_type']}")
        print(f"✓ Features: {model_info['feature_count']}")
        print(f"✓ Training Date: {model_info['training_date']}")
        print(f"✓ PR-AUC: {model_info['metrics']['pr_auc']:.4f}")

    # Run demos
    try:
        demo_single_prediction(client)
        demo_batch_prediction(client)
        demo_performance_test(client, num_requests=50)
        demo_monitoring(client)

        print("\n" + "=" * 60)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")

"""
USAGE:

1. Start the API:
   python app.py

2. Run demo in another terminal:
   python api_client_demo.py

3. Watch real-time predictions and performance metrics

EXPECTED OUTPUT:
- Single predictions with <50ms latency
- Batch processing of 10+ robots
- Performance test showing p95 < 50ms
- Real-time monitoring simulation
"""