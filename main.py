#!/usr/bin/env python3
"""
CNC G-code Sender Application

A modular, extensible CNC controller with visual cut preview capabilities.
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.application import main

if __name__ == "__main__":
    main()