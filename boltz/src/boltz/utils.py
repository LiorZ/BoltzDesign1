from pathlib import Path

import torch
from Bio.PDB import PDBParser, MMCIFParser


def load_ca_tensor(structure_file: str | Path, chain_id: str = "A", device: str | torch.device = "cpu") -> torch.Tensor:
    """Load CA coordinates from a PDB or CIF file as a tensor."""
    path = Path(structure_file).expanduser()
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("guide", path)
    model = structure[0]
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found in {path}")
    coords = [res["CA"].coord for res in model[chain_id] if "CA" in res]
    return torch.tensor(coords, dtype=torch.float32, device=device)
