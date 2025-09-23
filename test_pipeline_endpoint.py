#!/usr/bin/env python3
"""
Test script for the /pipeline endpoint.
This script tests the pipeline functionality that orchestrates predict, classify, and extract operations.
"""

import requests
import json
import sys
import argparse

def test_pipeline_endpoint(base_url="http://localhost:8888", image_path="Images/img1.png"):
    """Test the pipeline endpoint with sample data."""
    
    # Pipeline endpoint URL
    pipeline_url = f"{base_url}/pipeline/"
    
    # Test payload - you'll need to adjust these model IDs based on your actual configuration
    test_payload = {
        "predict_model_id": "yolov8n",  # Adjust based on your predict model
        "classify_model_id": "efficientnetv2",  # Adjust based on your classify model  
        "extract_model_id": "miewid",  # Adjust based on your extract model
        "image_uri": image_path,
        "bbox_score_threshold": 0.5,
        "predict_model_params": {
            "conf": 0.25,  # Optional: override default confidence threshold
            "iou": 0.45    # Optional: override default IoU threshold
        }
    }
    
    print(f"Testing pipeline endpoint: {pipeline_url}")
    print(f"Using image: {image_path}")
    print(f"Payload: {json.dumps(test_payload, indent=2)}")
    print("-" * 50)
    
    try:
        # Make the request
        response = requests.post(pipeline_url, json=test_payload, timeout=60)
        
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Pipeline request successful!")
            print(f"Total predictions: {result.get('total_predictions', 'N/A')}")
            print(f"Filtered predictions: {result.get('filtered_predictions', 'N/A')}")
            print(f"Pipeline results count: {len(result.get('pipeline_results', []))}")
            
            # Print summary of each pipeline result
            for i, pipeline_result in enumerate(result.get('pipeline_results', [])):
                print(f"\nBbox {i}:")
                print(f"  - Prediction score: {pipeline_result.get('prediction_score', 'N/A')}")
                print(f"  - Prediction class: {pipeline_result.get('prediction_class', 'N/A')}")
                print(f"  - Bbox coordinates: {pipeline_result.get('prediction_bbox', 'N/A')}")
                
                # Classification results
                classify_results = pipeline_result.get('classification_results', {})
                if 'predictions' in classify_results:
                    top_class = classify_results['predictions'][0] if classify_results['predictions'] else {}
                    print(f"  - Top classification: {top_class.get('class', 'N/A')} ({top_class.get('probability', 'N/A')})")
                
                # Extraction results
                extract_results = pipeline_result.get('extraction_results', {})
                embeddings_shape = extract_results.get('embeddings_shape', [])
                print(f"  - Embeddings shape: {embeddings_shape}")
            
            print(f"\n📄 Full response saved to pipeline_response.json")
            with open("pipeline_response.json", "w") as f:
                json.dump(result, f, indent=2)
                
        else:
            print(f"❌ Pipeline request failed!")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Error response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_models_endpoint(base_url="http://localhost:8888"):
    """Test the /predict/models endpoint to see available models."""
    
    models_url = f"{base_url}/predict/models"
    print(f"Testing models endpoint: {models_url}")
    
    try:
        response = requests.get(models_url, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print("✅ Available models:")
            for model_id, model_info in models.items():
                print(f"  - {model_id}: {model_info.get('handler_type', 'Unknown type')}")
            return models
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting models: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the pipeline endpoint')
    parser.add_argument('--base-url', default='http://localhost:8888', 
                       help='Base URL of the API server')
    parser.add_argument('--image', default='Images/img1.png',
                       help='Path to the test image')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models first')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("=" * 60)
        print("LISTING AVAILABLE MODELS")
        print("=" * 60)
        models = test_models_endpoint(args.base_url)
        print()
    
    print("=" * 60)
    print("TESTING PIPELINE ENDPOINT")
    print("=" * 60)
    test_pipeline_endpoint(args.base_url, args.image)
