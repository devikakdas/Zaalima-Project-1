"""
FactoryGuard AI - Production Flask API
Real-time prediction API with SHAP explainability and <50ms latency
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import shap
from datetime import datetime
import time
import logging
from feature_engineering import TimeSeriesFeatureEngineer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model artifacts (loaded once at startup)
MODEL = None
FEATURE_ENGINEER = None
METADATA = None
EXPLAINER = None
FEATURE_NAMES = None


def load_models():
    """Load all model artifacts at startup"""
    global MODEL, FEATURE_ENGINEER, METADATA, EXPLAINER, FEATURE_NAMES

    try:
        logger.info("Loading model artifacts...")

        # Load trained model
        MODEL = joblib.load('models/lightgbm_model.joblib')

        # Load feature engineering pipeline
        FEATURE_ENGINEER = joblib.load('models/feature_engineer.joblib')

        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            METADATA = json.load(f)

        FEATURE_NAMES = METADATA['feature_names']

        # Initialize SHAP explainer (pre-compute for speed)
        logger.info("Initializing SHAP explainer...")
        EXPLAINER = shap.TreeExplainer(MODEL)

        logger.info("✓ All model artifacts loaded successfully")
        logger.info(f"✓ Model type: {METADATA['model_type']}")
        logger.info(f"✓ Features: {len(FEATURE_NAMES)}")
        logger.info(f"✓ Optimal threshold: {METADATA['optimal_threshold']}")

    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Real-time prediction endpoint with explainability

    Expected request:
    {
        "robot_id": "ARM_247",
        "sensor_readings": {
            "vibration": [0.45, 0.48, ...],  # Last 12+ hours
            "temperature": [72.3, 74.1, ...],
            "pressure": [145.2, 146.0, ...]
        },
        "timestamp": "2026-01-06T14:30:00Z"
    }
    """
    start_time = time.time()

    try:
        # Parse request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        robot_id = data.get('robot_id')
        sensor_readings = data.get('sensor_readings')
        timestamp_str = data.get('timestamp')

        # Validate inputs
        if not all([robot_id, sensor_readings, timestamp_str]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Convert sensor readings to DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamp_str),
            'robot_id': robot_id,
            'vibration': sensor_readings['vibration'][-12:],  # Last 12 hours
            'temperature': sensor_readings['temperature'][-12:],
            'pressure': sensor_readings['pressure'][-12:]
        })

        # Ensure we have enough data points
        if len(df) < 12:
            return jsonify({'error': 'Need at least 12 hours of sensor data'}), 400

        # Feature engineering
        df_features = FEATURE_ENGINEER.transform(df)

        # Get latest reading features
        X = df_features[FEATURE_NAMES].iloc[-1:].values

        # Prediction
        failure_probability = float(MODEL.predict_proba(X)[0, 1])
        threshold = METADATA['optimal_threshold']
        risk_level = 'HIGH' if failure_probability >= threshold else 'MEDIUM' if failure_probability >= 0.3 else 'LOW'

        # SHAP explanation
        shap_values = EXPLAINER.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        # Top contributing factors
        feature_importance = []
        for i, feat_name in enumerate(FEATURE_NAMES):
            feature_importance.append({
                'feature': feat_name,
                'importance': float(abs(shap_values[0][i]))
            })

        # Sort by importance and take top 5
        feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:5]

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Prediction response
        response = {
            'robot_id': robot_id,
            'timestamp': timestamp_str,
            'failure_probability': round(failure_probability, 4),
            'risk_level': risk_level,
            'decision_threshold': threshold,
            'predicted_failure_time': None,  # Would be timestamp + 24h if HIGH risk
            'top_contributing_factors': feature_importance,
            'response_time_ms': round(response_time_ms, 2)
        }

        # Add predicted failure time if HIGH risk
        if risk_level == 'HIGH':
            failure_time = pd.to_datetime(timestamp_str) + pd.Timedelta(hours=24)
            response['predicted_failure_time'] = failure_time.isoformat()

        # Log prediction
        logger.info(json.dumps({
            'event': 'prediction',
            'robot_id': robot_id,
            'probability': failure_probability,
            'risk_level': risk_level,
            'latency_ms': response_time_ms
        }))

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information and metadata"""
    return jsonify({
        'model_type': METADATA['model_type'],
        'training_date': METADATA['training_date'],
        'feature_count': METADATA['feature_count'],
        'metrics': METADATA['metrics'],
        'optimal_threshold': METADATA['optimal_threshold']
    }), 200


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple robots
    Optimized for processing multiple predictions simultaneously
    """
    start_time = time.time()

    try:
        data = request.get_json()
        predictions = []

        for robot_data in data.get('robots', []):
            # Reuse single prediction logic
            with app.test_request_context(
                    '/predict',
                    method='POST',
                    json=robot_data
            ):
                response = predict()
                if response[1] == 200:
                    predictions.append(response[0].get_json())

        total_time_ms = (time.time() - start_time) * 1000

        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'total_time_ms': round(total_time_ms, 2),
            'avg_time_per_prediction_ms': round(total_time_ms / max(len(predictions), 1), 2)
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Load models at startup
load_models()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("FACTORYGUARD AI - PRODUCTION API")
    print("=" * 60)
    print("✓ Model loaded and ready")
    print("✓ SHAP explainer initialized")
    print(f"✓ Endpoints available:")
    print("  - GET  /health          (Health check)")
    print("  - POST /predict         (Single prediction)")
    print("  - POST /batch_predict   (Batch predictions)")
    print("  - GET  /model/info      (Model metadata)")
    print("\n✓ Starting Flask server...")
    print("=" * 60 + "\n")

    # Run in production mode with gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:5000 app:app

    app.run(host='0.0.0.0', port=5000, debug=False)