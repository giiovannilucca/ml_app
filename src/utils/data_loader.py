import pandas as pd
import base64

from typing import Optional
from pathlib import Path

def load_image_base64(relative_path: str) -> str:
    """
    Loads an image from a relative path and encodes it in base64 format.

    Parameters:
    - relative_path (str): Relative path to the image file.

    Returns:
    - str: Base64-encoded image string.
    """
    image_path = Path(__file__).resolve().parent.parent.parent / relative_path
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

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