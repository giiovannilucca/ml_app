import pandas as pd
import base64

from typing import Optional

def load_image_as_base64(path: str) -> str:
    """
    Loads an image from a given path and encodes it as a base64 string.

    Parameters:
    - path: Path to the image file (e.g., PNG, JPG).

    Returns:
    - Base64-encoded string of the image content.
    """
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

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