#!/bin/bash

# Test MSV3 model with image URL
echo "Testing with image URL..."
curl -X POST "http://localhost:6050/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "msv3",
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Green_sea_turtle_%28Chelonia_mydas%29_Moorea.jpg/500px-Green_sea_turtle_%28Chelonia_mydas%29_Moorea.jpg"
  }'

# echo "\nTesting with local image path..."
# # Test with local image path
# curl -X POST "http://localhost:6050/predict" \
#   -H "accept: application/json" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model_id": "msv3",
#     "image_path": "/app/model_config.json"  # This should be a real image path inside the container
#   }'

# echo "\nTesting with uploaded file..."
# # Test with file upload
# curl -X POST "http://localhost:6050/predict" \
#   -F "model_id=msv3" \
#   -F "file=@/path/to/your/local/image.jpg"  # Replace with a real path to your test image

# Performance test with specified number of requests
N=${1:-2}  # Use first argument if provided, otherwise default to 2
echo "\nRunning performance test with $N requests..."
URL="http://localhost:6050/predict"

time (
  for ((i = 1; i <= N; i++)); do
    echo "Sending request $i..."
    curl -X POST "$URL" \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -d '{"model_id": "msv3", "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Green_sea_turtle_%28Chelonia_mydas%29_Moorea.jpg/500px-Green_sea_turtle_%28Chelonia_mydas%29_Moorea.jpg"}' &
  done
  wait
)
