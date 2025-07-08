from ase.io import read
from pathlib import Path

_structures_dir = Path(__file__).parent

def load_structure(prefix):
    """Load .xyz or .traj file of tutorial structures by prefix from 'falcon/structures' directory (without extension)."""
    
    for ext in [".xyz", ".traj"]:
        path = _structures_dir / f"{prefix}{ext}"
        if path.exists():
            print(f"Structure {path} loaded.")
            return read(path)
    raise FileNotFoundError(f"No structure found for '{prefix}' in {_structures_dir}")
