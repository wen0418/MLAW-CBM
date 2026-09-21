#!/usr/bin/env python3
"""Train MLAW-CBM-Energy on the controlled scalp protocol."""

from .common import main_for_variant


if __name__ == "__main__":
    main_for_variant("mlaw_energy", __file__)
