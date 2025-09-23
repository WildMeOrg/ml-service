#!/usr/bin/env python3
"""
Test script for checkpoint loading functionality.
"""

import requests
import json
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
MODELS_ENDPOINT = f"{BASE_URL}/predict/models"

def test_checkpoint_loading():
    """Test that models with checkpoint_path are loaded correctly."""
    
    print("Testing checkpoint loading functionality...")
    print("=" * 50)
    
    try:
        response = requests.get(MODELS_ENDPOINT, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print("✅ Successfully retrieved models list")
            
            # Check for models with checkpoint loading
            checkpoint_models = []
            huggingface_models = []
            
            for model_id, info in models.items():
                model_info = info.get('info', {})
                use_checkpoint = model_info.get('use_checkpoint', False)
                checkpoint_path = model_info.get('checkpoint_path')
                
                print(f"\nModel: {model_id}")
                print(f"  Type: {info.get('model_type', 'Unknown')}")
                print(f"  Uses checkpoint: {use_checkpoint}")
                print(f"  Checkpoint path: {checkpoint_path}")
                print(f"  Version: {model_info.get('version', 'Unknown')}")
                
                if use_checkpoint:
                    checkpoint_models.append(model_id)
                else:
                    huggingface_models.append(model_id)
            
            print(f"\n📊 Summary:")
            print(f"  Models using checkpoints: {len(checkpoint_models)}")
            print(f"  Models using HuggingFace: {len(huggingface_models)}")
            
            if checkpoint_models:
                print(f"  Checkpoint models: {', '.join(checkpoint_models)}")
            if huggingface_models:
                print(f"  HuggingFace models: {', '.join(huggingface_models)}")
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        print("Make sure the ML service is running on localhost:8000")

def test_extract_with_checkpoint():
    """Test extraction endpoint with checkpoint-loaded model."""
    
    # Only test if we have a checkpoint model available
    extract_endpoint = f"{BASE_URL}/extract/"
    
    test_request = {
        "model_id": "miewid-msv3_checkpoint",  # Use checkpoint model if available
        "image_uri": "/data0/lasha.otarashvili/docker/ml-service/Images/img1.png",
        "bbox": [50, 50, 200, 200],
        "theta": 0.0
    }
    
    print(f"\nTesting extraction with checkpoint model...")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    
    try:
        response = requests.post(extract_endpoint, json=test_request, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Checkpoint model extraction successful!")
            print(f"Model ID: {result['model_id']}")
            print(f"Embeddings shape: {result['embeddings_shape']}")
        elif response.status_code == 404:
            print("\n⚠️  Checkpoint model not available - this is expected if no checkpoint file exists")
        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")

if __name__ == "__main__":
    print("MiewID Checkpoint Loading Test")
    print("=" * 40)
    
    # Test model loading
    test_checkpoint_loading()
    
    # Test extraction with checkpoint
    test_extract_with_checkpoint()
