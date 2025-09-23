#!/usr/bin/env python3
"""
Test script for the optional bbox functionality in the MiewID embeddings extraction endpoint.
"""

import requests
import json
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
EXTRACT_ENDPOINT = f"{BASE_URL}/extract/"

def test_extract_with_bbox():
    """Test the extract endpoint with bbox parameter."""
    
    test_request = {
        "model_id": "miewid_v3",
        "image_uri": "/data0/lasha.otarashvili/docker/ml-service/Images/img1.png",
        "bbox": [50, 50, 200, 200],  # x, y, width, height
        "theta": 0.0
    }
    
    print("Testing extract endpoint WITH bbox...")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    
    try:
        response = requests.post(EXTRACT_ENDPOINT, json=test_request, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success with bbox!")
            print(f"Embeddings shape: {result['embeddings_shape']}")
            print(f"Bbox used: {result['bbox']}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_extract_without_bbox():
    """Test the extract endpoint without bbox parameter (should use full image)."""
    
    test_request = {
        "model_id": "miewid_v3",
        "image_uri": "/data0/lasha.otarashvili/docker/ml-service/Images/img1.png",
        "theta": 0.0
    }
    
    print("\nTesting extract endpoint WITHOUT bbox (full image)...")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    
    try:
        response = requests.post(EXTRACT_ENDPOINT, json=test_request, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success without bbox!")
            print(f"Embeddings shape: {result['embeddings_shape']}")
            print(f"Bbox in response: {result['bbox']} (should be null)")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_models_endpoint():
    """Test the models endpoint to see available models."""
    models_endpoint = f"{BASE_URL}/predict/models"
    
    print("Checking available models...")
    
    try:
        response = requests.get(models_endpoint, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print("Available models:")
            for model_id, info in models.items():
                print(f"  - {model_id}: {info.get('handler_type', 'Unknown type')}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("Optional BBox Test for MiewID Extract Endpoint")
    print("=" * 50)
    
    # Check available models first
    if not test_models_endpoint():
        print("Cannot proceed without knowing available models")
        sys.exit(1)
    
    # Test both scenarios
    success_with_bbox = test_extract_with_bbox()
    success_without_bbox = test_extract_without_bbox()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"✅ Extract with bbox: {'PASS' if success_with_bbox else 'FAIL'}")
    print(f"✅ Extract without bbox: {'PASS' if success_without_bbox else 'FAIL'}")
    
    if success_with_bbox and success_without_bbox:
        print("\n🎉 All tests passed! Optional bbox functionality is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the service and try again.")
        sys.exit(1)
