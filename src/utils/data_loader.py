import pandas as pd
import base64
import os

from typing import Optional

def load_image_as_base64(relative_path: str) -> str:
    """
    Loads an image from a relative path and encodes it in base64 format.

    Parameters:
    - relative_path (str): Relative path to the image file.

    Returns:
    - str: Base64-encoded image string.
    """
    abs_path = os.path.join(os.path.dirname(__file__), "..", "..", relative_path)
    with open(abs_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

def load_csv(file) -> Optional[pd.DataFrame]:
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
    - file: Uploaded file object from Streamlit's file_uploader.

    Returns:
    - DataFrame if successful, otherwise None.
    """
    try:
        return pd.read_csv(file)
    except Exception:
        return None