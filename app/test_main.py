from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)

def test_pairx():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 200
    assert response.json()["output"]

    #add more good test cases, how?

def test_bad_path1():
    response = client.post(
        "/",
        json={"path1": "img1", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid path for image 1"}

def test_bad_path2():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid path for image 1"}

def test_bad_bb1():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3], "theta1": "0.0", 
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid bounding box for img1"}

    #add more ways for it to be bad (negative values, values greater than size of image, values that make box not work, etc.)

def test_bad_bb2():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid bounding box for img2"}

def test_bad_theta1():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "a", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Theta1 must be an integer"} #the actual error it will give is probably different

    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "-0.1", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Theta1 must be  positive"}

def test_bad_theta1():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "a",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Theta1 must be an integer"} #the actual error it will give is probably different

    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "-0.1",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Theta2 must be  positive"}

def test_bad_model():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "unknown", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Unknown model"}

def test_bad_visualization_type():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "unknown", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Unknown visualization type"}

def test_bad_layer_key():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "asdf", 
               "k_lines": 20, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Uknown layer key"}

def test_bad_k_lines():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": -1, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "k lines must be positive"}

    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 1000, "k_colors": 10},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "k lines must be less than x"}

def test_bad_k_colors():
    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 10, "k_colors": -1},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "k colors must be positive"}

    response = client.post(
        "/",
        json={"path1": "img1.png", "name1": "img1", "bb1": [1, 2, 3, 4], "theta1": "0.0",
               "path2": "img2.png", "name2": "img2", "bb2": [2, 3, 4, 5],
               "theta2": "0.0", "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 10, "k_colors": 1000},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "k colors must be less than x"}
