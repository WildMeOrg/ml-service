#!/usr/bin/env python3
"""
Test script for the MiewID embeddings extraction endpoint.
"""

import requests
import json
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
EXTRACT_ENDPOINT = f"{BASE_URL}/extract/"

def test_extract_endpoint():
    """Test the extract endpoint with a sample image."""
    
    # Test data
    test_request = {
        "model_id": "miewid_v3",  # Adjust this based on your model configuration
        "image_uri": "/data0/lasha.otarashvili/docker/ml-service/Images/img1.png",  # Using local test image
        "bbox": [50, 50, 200, 200],  # x, y, width, height
        "theta": 0.0  # No rotation
    }
    
    print("Testing MiewID embeddings extraction endpoint...")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    
    try:
        response = requests.post(EXTRACT_ENDPOINT, json=test_request, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print(f"Model ID: {result['model_id']}")
            print(f"Embeddings shape: {result['embeddings_shape']}")
            print(f"Bounding box: {result['bbox']}")
            print(f"Theta: {result['theta']}")
            print(f"Embeddings (first 5 values): {result['embeddings'][0][:5] if result['embeddings'] else 'None'}")
            
        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        print("Make sure the ML service is running on localhost:8000")

def test_models_endpoint():
    """Test the models endpoint to see available models."""
    models_endpoint = f"{BASE_URL}/predict/models"
    
    print("\nTesting models endpoint...")
    
    try:
        response = requests.get(models_endpoint, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print("✅ Available models:")
            for model_id, info in models.items():
                print(f"  - {model_id}: {info.get('handler_type', 'Unknown type')}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("MiewID Extract Endpoint Test")
    print("=" * 40)
    
    # First check available models
    test_models_endpoint()
    
    # Then test the extract endpoint
    test_extract_endpoint()
