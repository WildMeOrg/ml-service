#!/usr/bin/env python3
"""
Test script for the /classify endpoint.
Tests EfficientNet image classification functionality.
"""

import requests
import json
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8888"
CLASSIFY_ENDPOINT = f"{BASE_URL}/classify/"

def test_classify_endpoint():
    """Test the /classify endpoint with various scenarios."""
    
    # Test cases
    test_cases = [
        {
            "name": "Basic classification (local image)",
            "payload": {
                "model_id": "efficientnet-classifier",
                "image_uri": "/data0/lasha.otarashvili/docker/ml-service/Images/img1.png"
            }
        },
        {
            "name": "Classification with bounding box",
            "payload": {
                "model_id": "efficientnet-classifier",
                "image_uri": "/data0/lasha.otarashvili/docker/ml-service/Images/img2.png",
                "bbox": [100, 100, 200, 200]
            }
        },
        {
            "name": "Classification with rotation",
            "payload": {
                "model_id": "efficientnet-classifier",
                "image_uri": "/data0/lasha.otarashvili/docker/ml-service/Images/img3.png",
                "theta": 0.5
            }
        },
        {
            "name": "Classification with bbox and rotation",
            "payload": {
                "model_id": "efficientnet-classifier",
                "image_uri": "/data0/lasha.otarashvili/docker/ml-service/Images/img1.png",
                "bbox": [50, 50, 300, 300],
                "theta": 0.25
            }
        }
    ]
    
    print("Testing /classify endpoint...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            # Make the request
            response = requests.post(
                CLASSIFY_ENDPOINT,
                json=test_case["payload"],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"Model ID: {result.get('model_id')}")
                print(f"Predictions: {len(result.get('predictions', []))} classes above threshold")
                
                # Print top predictions
                predictions = result.get('predictions', [])
                if predictions:
                    print("Top predictions:")
                    for pred in predictions[:3]:  # Show top 3
                        print(f"  - {pred['label']}: {pred['probability']:.4f}")
                else:
                    print("No predictions above threshold")
                    
                print(f"Threshold: {result.get('threshold')}")
                
            else:
                print("❌ Failed!")
                try:
                    error_detail = response.json()
                    print(f"Error: {error_detail}")
                except:
                    print(f"Error: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the server is running on localhost:8888")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")

def test_error_cases():
    """Test error handling scenarios."""
    
    print("\nTesting error cases...")
    print("=" * 50)
    
    error_test_cases = [
        {
            "name": "Invalid model ID",
            "payload": {
                "model_id": "non-existent-model",
                "image_uri": "/data0/lasha.otarashvili/docker/ml-service/Images/img1.png"
            },
            "expected_status": 404
        },
        {
            "name": "Invalid bbox format",
            "payload": {
                "model_id": "efficientnet-classifier",
                "image_uri": "/data0/lasha.otarashvili/docker/ml-service/Images/img1.png",
                "bbox": [100, 100, 200]  # Missing one coordinate
            },
            "expected_status": 400
        },
        {
            "name": "Non-existent image file",
            "payload": {
                "model_id": "efficientnet-classifier",
                "image_uri": "/path/to/non/existent/image.jpg"
            },
            "expected_status": 400
        }
    ]
    
    for i, test_case in enumerate(error_test_cases, 1):
        print(f"\nError Test {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                CLASSIFY_ENDPOINT,
                json=test_case["payload"],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == test_case["expected_status"]:
                print("✅ Expected error handled correctly!")
            else:
                print(f"❌ Unexpected status code. Expected: {test_case['expected_status']}")
            
            try:
                error_detail = response.json()
                print(f"Error detail: {error_detail.get('detail', 'No detail')}")
            except:
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

def check_server_status():
    """Check if the server is running and accessible."""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ Server is running and accessible")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running on localhost:8888")
        return False
    except Exception as e:
        print(f"❌ Error checking server status: {str(e)}")
        return False

if __name__ == "__main__":
    print("EfficientNet Classification Endpoint Test")
    print("=" * 50)
    
    # Check server status first
    if not check_server_status():
        print("\nPlease start the server first:")
        print("python -m app.main --host 0.0.0.0 --port 8888")
        sys.exit(1)
    
    # Run tests
    test_classify_endpoint()
    test_error_cases()
    
    print("\n🎉 All tests completed!")
    print("\nTo run individual tests, you can also use curl:")
    print("curl -X POST http://localhost:8888/classify/ \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{")
    print('    "model_id": "efficientnet-classifier",')
    print('    "image_uri": "/path/to/your/image.jpg"')
    print("  }'")
