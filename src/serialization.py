import joblib
import pandas as pd
from typing import Any
from pathlib import Path
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None




def save_joblib(model: Any, path: Path) -> None:
    """Persist model with joblib."""
    joblib.dump(model, path)

def load_joblib(path: Path) -> Any:
    """Load persisted model."""
    return joblib.load(path)

def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist.
    Used to ensure output directories are present before saving files."""
    path.mkdir(parents=True, exist_ok=True)

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV."""
    df.to_csv(path, index=False)

def load_dataframe(path: Path) -> pd.DataFrame:
    """Load DataFrame from CSV."""
    return pd.read_csv(path)

def save_figure(fig: Any, path: Path) -> None:
    """Save figure to file."""
    fig.savefig(path)
    fig.clf()

def load_figure(path: Path) -> Any:
    """Load a raster image saved by ``save_figure`` (e.g. PNG). Returns an ndarray (RGB/A)."""
    import matplotlib.image as mpimg

    return mpimg.imread(str(path))

def save_numpy(array: Any, path: Path) -> None:
    """Save NumPy array to file."""
    np.save(path, array)
def save_multiple_numpy(arrays: dict[str, Any], path: Path) -> None:
    """Save multiple NumPy arrays to a single file (e.g. with np.savez)."""
    np.savez(path, **arrays)

def load_numpy(path: Path) -> Any:
    """Load NumPy array from file."""
    return np.load(path)

def save_hdf5_dict(data: dict[str, Any], path: Path) -> None:
    """Save dictionary to HDF5 file with append mode.
    
    Handles numpy arrays and lists. Scalar values are stored as datasets.
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        for key, value in data.items():
            if key in f:
                del f[key]  # Overwrite existing key
            
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            elif isinstance(value, (list, tuple)):
                f.create_dataset(key, data=np.array(value))
            elif isinstance(value, (str, int, float, bool)):
                f.create_dataset(key, data=value)
            else:
                # Fallback: try to convert to numpy
                f.create_dataset(key, data=np.array(value))

def load_hdf5_dict(path: Path) -> dict[str, Any]:
    """Load dictionary from HDF5 file.
    
    Returns numpy arrays for array data, scalars for scalar data.
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    
    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            value = f[key][()]
            data[key] = value
    return data