#!/bin/bash

URL="http://localhost:8000/predict/?model_id=model1&image_url=https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR73V4nXNazAtkt4peyPOjzhJsFRXBtLVQHDRiYS0r3TA&s&ec=72940544"
N=${1:-5}  # Default to 5 if no argument provided

time (
  for ((i = 1; i <= N; i++)); do
    curl -s -X POST "$URL" &
  done
  wait
)
