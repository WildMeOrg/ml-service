from fastapi.testclient import TestClient

from .main import app


def test_pairx():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={
                "image1_uris": ["Images/img1.png"],
                "bb1": [
                    [0, 0, 0, 0]
                ],
                "image2_uris": ["Images/img2.png"],
                "bb2": [
                    [0, 0, 0, 0]
                ],
                "model_id": "miewid-msv3",
                "crop_bbox": False,
                "visualization_type": "lines_and_colors",
                "layer_key": "backbone.blocks.3",
                "k_lines": 20,
                "k_colors": 5
            }

        )
        assert response.status_code == 200
        assert response.json()["response"]
    
        response = client.post(
            "/explain/",
            json={
                "image1_uris": ["Images/img1.png", "Images/img1.png"],
                "bb1": [
                    [0, 0, 0, 0]
                ],
                "image2_uris": ["Images/img2.png", "Images/img1.png"],
                "bb2": [
                    [0, 0, 0, 0]
                ],
                "model_id": "miewid-msv3",
                "crop_bbox": False,
                "visualization_type": "lines_and_colors",
                "layer_key": "backbone.blocks.3",
                "k_lines": 20,
                "k_colors": 5
            }
        )
        assert response.status_code == 200
        assert response.json()["response"]

def test_bad_path1():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={"image1_uris": ["img1"], "bb1": [[1, 2, 3, 4]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "bb2": [[2, 3, 4, 5]],
               "theta2": [0.0], "model_id": "miewid-msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
        )
        assert response.status_code == 400
        assert response.json()["detail"].startswith("Error loading image")

def test_bad_bb1():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "bb1": [[1, 2, 3]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "bb2": [[2, 3, 4, 5]], "theta2": [0.0],
               "model_id": "miewid-msv3", "crop_bbox": False,
               "visualization_type": "lines_and_colors",
               "layer_key": "backbone.blocks.3",
               "k_lines": 20, "k_colors": 5},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Each bounding box should have 4 values"}

def test_bad_bb2():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "bb1": [[1, 2, 3, 4]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "bb2": [[2, 3, 4]], "theta2": [0.0],
               "model_id": "miewid-msv3", "crop_bbox": False,
               "visualization_type": "lines_and_colors",
               "layer_key": "backbone.blocks.3",
               "k_lines": 20, "k_colors": 5},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "400: Each bounding box should have 4 values"

def test_bad_theta1():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "bb1": [[1, 2, 3, 4]], "theta1": [-10.0],
               "image2_uris": ["Images/img2.png"], "bb2": [[2, 3, 4, 5]], "theta2": [0.0],
               "model_id": "miewid-msv3", "crop_bbox": False,
               "visualization_type": "lines_and_colors",
               "layer_key": "backbone.blocks.3",
               "k_lines": 20, "k_colors": 5},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Theta should be greater than 0"}

def test_bad_theta2():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "bb1": [[1, 2, 3, 4]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "bb2": [[2, 3, 4, 5]],
               "theta2": [-10.0], "model_id": "miewid-msv3", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "400: Theta should be greater than 0"}

def test_bad_model():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "name1": "img1", "bb1": [[1, 2, 3, 4]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "name2": "img2", "bb2": [[2, 3, 4, 5]],
               "theta2": [0.0], "model_id": "unknown", "crop_bbox": False, 
               "visualization_type": "lines_and_colors", 
               "layer_key": "backbone.blocks.3", 
                  "k_lines": 20, "k_colors": 10, "algorithm": "pairx"},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Unsupported model for pairx."}

def test_bad_visualization_type():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "bb1": [[1, 2, 3, 4]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "name2": "img2", "bb2": [[2, 3, 4, 5]],
               "theta2": [0.0], "model_id": "msv3", "crop_bbox": False, 
               "visualization_type": "unknown", 
               "layer_key": "backbone.blocks.3", 
               "k_lines": 20, "k_colors": 10},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Unsupported visualization type."}

def test_bad_layer_key():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "bb1": [[1, 2, 3, 4]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "bb2": [[2, 3, 4, 5]], "theta2": [0.0],
               "model_id": "miewid-msv3", "crop_bbox": False,
               "visualization_type": "lines_and_colors",
               "layer_key": "backbone.blasdf",
               "k_lines": 20, "k_colors": 5},
            )
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid layer key"}

def test_bad_k_lines():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "bb1": [[1, 2, 3, 4]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "bb2": [[2, 3, 4, 5]], "theta2": [0.0],
               "model_id": "miewid-msv3", "crop_bbox": False,
               "visualization_type": "lines_and_colors",
               "layer_key": "backbone.blocks.3",
               "k_lines": -3, "k_colors": 5},
            )
        assert response.status_code == 400
        assert response.json() == {"detail": "K Lines must be positive"}

        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "bb1": [[1, 2, 3, 4]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "bb2": [[2, 3, 4, 5]], "theta2": [0.0],
               "model_id": "miewid-msv3", "crop_bbox": False,
               "visualization_type": "lines_and_colors",
               "layer_key": "backbone.blocks.3",
               "k_lines": 1000, "k_colors": 5},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "K Lines must be less than 100"}

def test_bad_k_colors():
    with TestClient(app) as client:
        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "bb1": [[1, 2, 3, 4]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "bb2": [[2, 3, 4, 5]], "theta2": [0.0],
               "model_id": "miewid-msv3", "crop_bbox": False,
               "visualization_type": "lines_and_colors",
               "layer_key": "backbone.blocks.3",
               "k_lines": 10, "k_colors": -3},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "K Colors must be positive"}

        response = client.post(
            "/explain/",
            json={"image1_uris": ["Images/img1.png"], "bb1": [[1, 2, 3, 4]], "theta1": [0.0],
               "image2_uris": ["Images/img2.png"], "bb2": [[2, 3, 4, 5]], "theta2": [0.0],
               "model_id": "miewid-msv3", "crop_bbox": False,
               "visualization_type": "lines_and_colors",
               "layer_key": "backbone.blocks.3",
               "k_lines": 10, "k_colors": 100},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "K Colors must be less than 100"}
