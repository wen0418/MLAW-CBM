"""Public model package.

The historical package-level ``mvpcbm`` name now resolves to the stable
MLAW-CBM paper model.  Explicit baseline and ablation implementations remain
available from their own modules.
"""

from .mlaw_cbm import MLAWCBM, mvpcbm
from .mlaw_cbm_energy import MLAWCBMEnergy

__all__ = ["MLAWCBM", "MLAWCBMEnergy", "mvpcbm"]
