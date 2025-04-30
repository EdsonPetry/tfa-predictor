import numpy as np

def numpy_to_python(obj):
    """Convert NumPy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, bool):
        return bool(obj)  # Explicitly convert bool to ensure JSON serialization
    else:
        return obj