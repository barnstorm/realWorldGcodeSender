# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based CNC G-code sender application that provides visual cut preview functionality by overlaying G-code paths onto camera images of the CNC bed. The system uses computer vision (OpenCV) with ArUco/ChArUco markers to detect and map the physical CNC bed space to provide accurate cut path visualization.

## Key Dependencies

- **opencv-python (cv2)**: Computer vision for marker detection and image processing
- **numpy**: Numerical computations and array operations
- **matplotlib**: GUI interface and visualization
- **svgpathtools**: SVG path processing and manipulation
- **pygcode**: G-code parsing and machine state simulation
- **pyserial**: Serial communication with CNC controller
- **svgToGCode** (local dependency in ../svgToGCode/): Custom SVG to G-code conversion

## Architecture & Core Components

### Main Application (`realWorldGcodeSender.py`)
- Entry point and main GUI application
- Handles camera capture and image processing
- Manages ChArUco marker detection for bed calibration
- Implements homography transformation for pixel-to-physical coordinate mapping
- Provides interactive matplotlib-based UI for G-code manipulation

### CNC Communication Layer
- **`gerbil.py`**: High-level GRBL controller interface
  - Manages G-code streaming and buffering
  - Handles machine state tracking and error recovery
  - Implements probing and homing sequences
- **`interface.py`**: Low-level serial communication handler
- **`gcode_machine.py`**: G-code state machine and position tracking

### Coordinate System
- Machine origin: Right, back, upper corner (0,0,0)
- Bed size: 35" x 35" x 3.75" (negative coordinates from origin)
- ChArUco markers placed on vertical rails for calibration
- Uses homography to map camera pixels to physical CNC coordinates

## Development Commands

### Running the Application
```bash
python realWorldGcodeSender.py
```

### Testing Components
```bash
python test.py  # Basic functionality tests
```

## Important Configuration

### Physical Setup Constants (in `realWorldGcodeSender.py`)
- `boxWidth`: 0.745642857" - Width of ChArUco marker boxes
- `materialThickness`: 0.471" - Default material thickness
- `cutterDiameter`: 0.125" - Default cutter diameter
- `bedSize`: Point3D(-35.0, -35.0, -3.75) - CNC bed dimensions
- Marker reference positions defined in `rightBoxRef` and `leftBoxRef`

### Serial Communication
- Default baud rate: 115200
- Uses automatic COM port detection via `serial.tools.list_ports`

## Key Features & Workflows

### Camera Calibration & Image Processing
1. Captures image from camera (any angle supported)
2. Detects ChArUco markers on vertical rails
3. Applies homography transformation to map pixels to physical coordinates
4. Displays physical location (in inches) when clicking on image

### G-code Handling
- Loads and parses G-code files
- Simulates tool paths using pygcode Machine
- Allows interactive repositioning and rotation of G-code
- Supports both standard G-code and SVG import (via svgToGCode)

### Probing & Homing
- Auto-detects probe plate in camera image
- Performs XYZ probing sequence
- Sets work coordinate system based on probe results

## Code Style Notes

- Uses global variables for configuration constants
- Matplotlib event-driven GUI with callback handlers
- Threading used for serial communication and long-running operations
- Extensive use of numpy arrays for coordinate transformations

## Common Tasks

### Adding New Features
- GUI interactions: Add matplotlib event handlers in main loop
- G-code operations: Extend `gerbil.py` or `gcode_machine.py`
- Image processing: Modify detection logic in `realWorldGcodeSender.py`

### Debugging Serial Communication
- Check `gerbil.py` logging output
- Monitor serial interface in `interface.py`
- Verify GRBL responses and error codes

### Calibration Issues
- Verify ChArUco marker IDs match `idToLocDict` mapping
- Check marker physical positions match configuration
- Adjust `boxWidth` if marker spacing is incorrect