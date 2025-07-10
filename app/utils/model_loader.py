from ultralytics import YOLO

def load_model(model_path: str, device: str):
    """
    Load a YOLO model onto the specified device.

    Args:
        model_path (str): Path to the YOLO model file.
        device (str): Device to load the model on (e.g., 'cpu', 'cuda').

    Returns:
        model: Loaded YOLO model.
    """
    model = YOLO(model_path)
    model.to(device)
    return model