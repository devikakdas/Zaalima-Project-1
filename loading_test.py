"""
FactoryGuard AI - Load Testing
Simulate 500 concurrent robotic arms making predictions
Target: p95 latency < 50ms
"""

from locust import HttpUser, task, between
import json
import random
from datetime import datetime, timedelta


class RobotArmUser(HttpUser):
    """
    Simulates a robotic arm sending sensor data for prediction
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Initialize robot on startup"""
        self.robot_id = f"ARM_{random.randint(0, 499):03d}"

    def generate_sensor_data(self, failure_mode=False):
        """Generate realistic sensor readings"""
        if failure_mode:
            # Degradation pattern
            vibration = [random.uniform(0.5, 0.8) for _ in range(12)]
            temperature = [random.uniform(80, 95) for _ in range(12)]
            pressure = [random.uniform(170, 200) for _ in range(12)]
        else:
            # Normal operation
            vibration = [random.uniform(0.25, 0.35) for _ in range(12)]
            temperature = [random.uniform(60, 70) for _ in range(12)]
            pressure = [random.uniform(140, 160) for _ in range(12)]

        return {
            'robot_id': self.robot_id,
            'sensor_readings': {
                'vibration': vibration,
                'temperature': temperature,
                'pressure': pressure
            },
            'timestamp': (datetime.now() - timedelta(hours=12)).isoformat()
        }

    @task(9)
    def predict_normal(self):
        """Predict on normal sensor data (90% of requests)"""
        payload = self.generate_sensor_data(failure_mode=False)

        with self.client.post(
                "/predict",
                json=payload,
                catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                latency = data.get('response_time_ms', 0)

                # Check latency SLA
                if latency > 50:
                    response.failure(f"Latency {latency}ms exceeds 50ms SLA")
                else:
                    response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def predict_failure(self):
        """Predict on degradation pattern (10% of requests)"""
        payload = self.generate_sensor_data(failure_mode=True)

        with self.client.post(
                "/predict",
                json=payload,
                catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                latency = data.get('response_time_ms', 0)

                # Verify high risk detection
                if data.get('risk_level') != 'HIGH':
                    response.failure("Failed to detect degradation pattern")
                elif latency > 50:
                    response.failure(f"Latency {latency}ms exceeds 50ms SLA")
                else:
                    response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def health_check(self):
        """Periodic health check"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


"""
RUNNING LOAD TESTS:

1. Start the API:
   python app.py

2. Run load test:
   locust -f locustfile.py --host=http://localhost:5000

3. Open browser: http://localhost:8089

4. Test configuration:
   - Number of users: 500 (simulating 500 robots)
   - Spawn rate: 50 users/second
   - Run time: 5 minutes

5. Monitor metrics:
   - Request rate (should handle 500 concurrent)
   - p50, p95, p99 latency (p95 MUST be <50ms)
   - Failure rate (should be <1%)

EXPECTED RESULTS:
✓ p50 latency: ~15-25ms
✓ p95 latency: <50ms (SLA requirement)
✓ p99 latency: <100ms
✓ Throughput: 200+ req/sec
✓ Error rate: <0.1%
"""