#!/usr/bin/env python3
"""
Test script to verify the MiewID model checkpoint loading fix.
"""
import sys
import os
sys.path.append('/data0/lasha.otarashvili/docker/ml-service')

from app.models.miewid import MiewidModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_miewid_loading():
    """Test MiewID model loading with checkpoint_path parameter."""
    try:
        # Test case 1: Loading without checkpoint (HuggingFace mode)
        logger.info("Testing MiewID loading without checkpoint...")
        model1 = MiewidModel()
        model1.load(device="cpu", version=3, imgsz=440)
        logger.info("✓ Successfully loaded MiewID without checkpoint")
        
        # Test case 2: Loading with checkpoint_path (should not cause parameter conflict)
        logger.info("Testing MiewID loading with checkpoint_path...")
        model2 = MiewidModel()
        # This should not raise "multiple values for argument 'checkpoint_path'" error
        try:
            model2.load(
                device="cpu", 
                version=3, 
                imgsz=440, 
                checkpoint_path="/fake/path/checkpoint.bin"  # Fake path for testing parameter passing
            )
        except FileNotFoundError:
            # Expected since the checkpoint file doesn't exist
            logger.info("✓ Parameter passing works correctly (FileNotFoundError expected)")
        except TypeError as e:
            if "multiple values for argument" in str(e):
                logger.error("✗ Parameter conflict still exists!")
                raise
            else:
                # Some other TypeError, re-raise
                raise
        
        logger.info("All tests passed! The checkpoint_path parameter conflict has been fixed.")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_miewid_loading()
    sys.exit(0 if success else 1)
