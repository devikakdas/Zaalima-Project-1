#!/bin/bash

# Exit on error
set -e

URL="http://localhost:5000/predict"

echo "Sending prediction request to $URL"

curl -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "ARM_001",
    "sensor_readings": {
      "vibration": [0.30, 0.31, 0.29, 0.32, 0.30, 0.31, 0.30, 0.29, 0.31, 0.30, 0.32, 0.30],
      "temperature": [65, 66, 64, 67, 65, 66, 65, 64, 66, 65, 67, 65],
      "pressure": [150, 151, 149, 152, 150, 151, 150, 149, 151, 150, 152, 150]
    },
    "timestamp": "2026-01-08T13:30:00Z"
  }'

echo
echo "Request completed."
