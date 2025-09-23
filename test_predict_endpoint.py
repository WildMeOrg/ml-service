#!/usr/bin/env python3
"""
Sample request script for the prediction/detection endpoint.
"""

import requests
import time
import concurrent.futures
from pprint import pprint

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust if your service runs on different host/port
PREDICT_ENDPOINT = f"{BASE_URL}/predict/"

# Test image URLs and paths
image_url = "https://res.cloudinary.com/roundglass/image/upload/f_auto/q_auto/f_auto/c_limit,w_auto:breakpoints_200_2560_100_5:1265/v1572167551/roundglass/sustain/Spotted-Hyenas_-MicheleB_-Shutterstock_tu5ggi.jpg"
local_image_path = "/datasets/coco_ms1_1/coco/images/test2023/000000000001.jpg"

def test_yolo_detection():
    """Test YOLO detection with image URL."""
    print("=" * 60)
    print("Testing YOLO Detection with image URL...")
    print("=" * 60)
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    data = {
        "model_id": "yolov8n",
        "image_uri": image_url,
        "model_params": {
            "conf": 0.25,  # Confidence threshold
            "iou": 0.45,   # IoU threshold for NMS
            "imgsz": 640   # Image size
        }
    }
    
    print(f"Request URL: {PREDICT_ENDPOINT}")
    print(f"Request data:")
    pprint(data)
    print()
    
    try:
        response = requests.post(PREDICT_ENDPOINT, headers=headers, json=data, timeout=30)
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ YOLO Detection successful!")
            print(f"Model ID: {result.get('model_id')}")
            
            # Print detection results
            detections = result.get('detections', [])
            print(f"Number of detections: {len(detections)}")
            
            for i, detection in enumerate(detections[:5]):  # Show first 5 detections
                print(f"  Detection {i+1}:")
                print(f"    Class: {detection.get('class_name', 'Unknown')} (ID: {detection.get('class_id', 'N/A')})")
                print(f"    Confidence: {detection.get('confidence', 0):.3f}")
                print(f"    Bbox: {detection.get('bbox', [])}")
            
            if len(detections) > 5:
                print(f"    ... and {len(detections) - 5} more detections")
                
        else:
            print(f"❌ Error {response.status_code}")
            print("Response text:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_megadetector():
    """Test MegaDetector with local image path."""
    print("\n" + "=" * 60)
    print("Testing MegaDetector with local image...")
    print("=" * 60)
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    data = {
        "model_id": "megadetector_v5a",
        "image_uri": local_image_path,
        "model_params": {
            "conf": 0.1,    # Lower confidence for wildlife detection
            "iou": 0.45,
            "imgsz": 1280   # Higher resolution for MegaDetector
        }
    }
    
    print(f"Request URL: {PREDICT_ENDPOINT}")
    print(f"Request data:")
    pprint(data)
    print()
    
    try:
        response = requests.post(PREDICT_ENDPOINT, headers=headers, json=data, timeout=30)
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ MegaDetector successful!")
            print(f"Model ID: {result.get('model_id')}")
            
            # Print detection results
            detections = result.get('detections', [])
            print(f"Number of detections: {len(detections)}")
            
            for i, detection in enumerate(detections):
                print(f"  Detection {i+1}:")
                print(f"    Class: {detection.get('class_name', 'Unknown')} (ID: {detection.get('class_id', 'N/A')})")
                print(f"    Confidence: {detection.get('confidence', 0):.3f}")
                print(f"    Bbox: {detection.get('bbox', [])}")
                
        else:
            print(f"❌ Error {response.status_code}")
            print("Response text:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_with_custom_params():
    """Test with custom model parameters."""
    print("\n" + "=" * 60)
    print("Testing with custom parameters...")
    print("=" * 60)
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    data = {
        "model_id": "yolov8n",
        "image_uri": image_url,
        "model_params": {
            "conf": 0.5,    # Higher confidence threshold
            "iou": 0.3,     # Lower IoU threshold (more aggressive NMS)
            "imgsz": 320,   # Smaller image size for faster inference
            "max_det": 10   # Maximum detections
        }
    }
    
    print(f"Request data:")
    pprint(data)
    print()
    
    try:
        start_time = time.time()
        response = requests.post(PREDICT_ENDPOINT, headers=headers, json=data, timeout=30)
        end_time = time.time()
        
        print(f"Response Status: {response.status_code}")
        print(f"Inference time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Custom parameters test successful!")
            
            detections = result.get('detections', [])
            print(f"Number of detections: {len(detections)}")
            
        else:
            print(f"❌ Error {response.status_code}")
            print("Response text:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_concurrent_requests():
    """Test multiple concurrent requests."""
    print("\n" + "=" * 60)
    print("Testing concurrent requests...")
    print("=" * 60)
    
    def make_request(request_id):
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        data = {
            "model_id": "yolov8n",
            "image_uri": image_url,
            "model_params": {
                "conf": 0.25,
                "imgsz": 640
            }
        }
        
        start_time = time.time()
        try:
            response = requests.post(PREDICT_ENDPOINT, headers=headers, json=data, timeout=30)
            end_time = time.time()
            
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "time": end_time - start_time,
                "detections": len(response.json().get('detections', [])) if response.status_code == 200 else 0
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "error": str(e),
                "time": time.time() - start_time
            }
    
    # Make 3 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(make_request, i) for i in range(3)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    print("Concurrent request results:")
    for result in sorted(results, key=lambda x: x.get('request_id', 0)):
        if 'error' in result:
            print(f"  Request {result['request_id']}: ERROR - {result['error']} (Time: {result['time']:.2f}s)")
        else:
            print(f"  Request {result['request_id']}: Status {result['status_code']}, {result['detections']} detections (Time: {result['time']:.2f}s)")

if __name__ == "__main__":
    print("ML Service Prediction Endpoint Test")
    print("Make sure the ML service is running on localhost:8000")
    print("(or update BASE_URL in the script)")
    
    # Test different detection models and scenarios
    test_yolo_detection()
    test_megadetector()
    test_with_custom_params()
    test_concurrent_requests()
    
    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)
