#!/usr/bin/env python3
"""
Configuration launcher for realWorldGcodeSender

This script provides easy access to the configuration GUI.
"""

import sys
import os

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_gui import main

if __name__ == "__main__":
    print("Opening realWorldGcodeSender Configuration GUI...")
    main()