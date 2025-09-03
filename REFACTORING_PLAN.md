# CNC G-Code Sender Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring strategy for the realWorldGcodeSender project to transform it from a proof-of-concept into a maintainable, extensible production-ready application. The refactoring will enable implementation of desired features while improving code quality, performance, and user experience.

## Current State Analysis

### Problems with Existing Architecture

1. **Monolithic Structure**
   - Single 1449-line file (`realWorldGcodeSender.py`) containing all functionality
   - Mixed responsibilities: UI, vision processing, machine control, path processing
   - Difficult to test, maintain, and extend

2. **Global State Management**
   - Extensive use of global variables for configuration
   - Hard-coded physical dimensions and calibration values
   - No centralized configuration management

3. **Tight Coupling**
   - Direct matplotlib event handling mixed with business logic
   - Camera capture tied to specific implementation
   - Machine control embedded in UI code

4. **Limited Extensibility**
   - Cannot easily swap UI frameworks
   - No abstraction for different camera types
   - Hard to add new visualization modes (3D view)

5. **Performance Issues**
   - Python/matplotlib responsiveness problems noted by author
   - Single-threaded blocking operations
   - No optimization for real-time updates

## Proposed Architecture

### Design Principles

- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Dependency Injection**: Components receive dependencies rather than creating them
- **Interface Segregation**: Abstract interfaces for swappable implementations
- **Configuration as Code**: All settings externalized and validated
- **Event-Driven Architecture**: Loose coupling through event system
- **Plugin Architecture**: Support for extending functionality without modifying core

### Module Structure

```
realWorldGcodeSender/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration classes
│   ├── loader.py            # YAML/JSON config loader
│   ├── validator.py         # Config validation
│   └── defaults/
│       ├── machine.yaml     # Machine specifications
│       ├── camera.yaml      # Camera settings
│       └── cutting.yaml     # Cutting parameters
│
├── vision/
│   ├── __init__.py
│   ├── camera/
│   │   ├── base.py         # Abstract camera interface
│   │   ├── webcam.py       # Webcam implementation
│   │   ├── static.py       # Static image loader
│   │   └── stream.py       # Live stream handler
│   ├── calibration/
│   │   ├── marker_detector.py   # ChArUco detection
│   │   ├── homography.py        # Coordinate transformation
│   │   └── auto_calibrate.py    # Auto-calibration routines
│   └── detection/
│       ├── line_detector.py     # Pencil/marker line detection
│       ├── probe_finder.py      # Probe plate detection
│       └── edge_detector.py     # Workpiece edge detection
│
├── path/
│   ├── __init__.py
│   ├── processors/
│   │   ├── gcode.py        # G-code parsing and generation
│   │   ├── svg.py          # SVG import and conversion
│   │   └── drawing.py      # User-drawn path handling
│   ├── optimization/
│   │   ├── path_optimizer.py    # Path optimization algorithms
│   │   ├── tab_generator.py     # Automatic tab placement
│   │   └── toolpath.py          # Toolpath strategies
│   └── transformations/
│       ├── rotate.py        # Rotation operations
│       ├── translate.py     # Translation operations
│       └── scale.py         # Scaling operations
│
├── visualization/
│   ├── __init__.py
│   ├── renderers/
│   │   ├── base.py         # Abstract renderer
│   │   ├── matplotlib_2d.py # 2D matplotlib renderer
│   │   ├── opengl_3d.py    # 3D OpenGL renderer
│   │   └── web_canvas.py   # Web-based canvas renderer
│   ├── overlay/
│   │   ├── path_overlay.py  # Path overlay manager
│   │   ├── position_tracker.py # Current position display
│   │   └── cut_preview.py   # Cut simulation preview
│   └── themes/
│       └── default.py       # Visual theme definitions
│
├── machine/
│   ├── __init__.py
│   ├── controllers/
│   │   ├── base.py         # Abstract machine controller
│   │   ├── grbl.py         # GRBL-specific implementation
│   │   └── simulation.py   # Simulated machine for testing
│   ├── operations/
│   │   ├── probe.py        # Probing operations
│   │   ├── homing.py       # Homing sequences
│   │   ├── jogging.py      # Manual jogging control
│   │   └── execution.py    # G-code execution engine
│   └── state/
│       ├── position.py     # Position tracking
│       ├── status.py       # Machine status management
│       └── limits.py       # Soft/hard limit handling
│
├── ui/
│   ├── __init__.py
│   ├── interfaces/
│   │   ├── base.py         # Abstract UI interface
│   │   ├── matplotlib_ui.py # Current matplotlib implementation
│   │   ├── qt_ui.py        # PyQt/PySide implementation
│   │   └── web_ui.py       # Web-based interface
│   ├── components/
│   │   ├── camera_view.py  # Camera display widget
│   │   ├── path_editor.py  # Path editing controls
│   │   ├── machine_status.py # Status display
│   │   └── settings_dialog.py # Configuration UI
│   └── events/
│       ├── handler.py      # Event handling system
│       └── dispatcher.py   # Event dispatching
│
├── core/
│   ├── __init__.py
│   ├── application.py      # Main application class
│   ├── coordinator.py      # Module coordination
│   ├── event_bus.py        # Central event system
│   └── plugin_manager.py   # Plugin loading system
│
├── plugins/
│   ├── __init__.py
│   └── example_plugin.py   # Example plugin structure
│
├── utils/
│   ├── __init__.py
│   ├── geometry.py         # Geometric calculations
│   ├── units.py           # Unit conversion utilities
│   ├── logging_config.py  # Logging configuration
│   └── validators.py      # Input validation utilities
│
├── tests/
│   ├── unit/              # Unit tests for each module
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data and fixtures
│
├── docs/
│   ├── api/              # API documentation
│   ├── user_guide/       # User documentation
│   └── developer/        # Developer guide
│
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
└── README.md            # Project documentation
```

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Establish project structure and core abstractions

**Tasks**:
1. Create project directory structure
2. Set up configuration management system
3. Implement logging framework
4. Create abstract interfaces for camera, UI, and machine
5. Set up unit testing framework
6. Create event bus system

**Deliverables**:
- Working configuration loader with validation
- Abstract base classes for major components
- Event system with publish/subscribe
- Basic test suite structure

### Phase 2: Vision Module (Weeks 3-4)
**Goal**: Extract and modularize all vision-related functionality

**Tasks**:
1. Extract ChArUco marker detection code
2. Implement camera abstraction with webcam support
3. Create homography and coordinate transformation module
4. Add probe plate detection
5. Implement calibration storage and loading

**Deliverables**:
- Standalone vision module with clean API
- Support for multiple camera sources
- Persistent calibration data
- Unit tests for coordinate transformations

### Phase 3: Path Processing (Weeks 5-6)
**Goal**: Separate path processing and optimization logic

**Tasks**:
1. Extract G-code parsing and generation
2. Modularize SVG import functionality
3. Implement path optimization algorithms
4. Create tab generation system
5. Add path transformation operations

**Deliverables**:
- Clean path processing API
- Optimized toolpath generation
- Automatic tab placement
- Path manipulation tools

### Phase 4: Machine Control (Weeks 7-8)
**Goal**: Create abstracted machine control layer

**Tasks**:
1. Extract GRBL communication code
2. Implement machine state management
3. Create probing operation manager
4. Add position tracking system
5. Implement simulation mode for testing

**Deliverables**:
- Machine-agnostic control interface
- Robust error handling
- Simulation mode for development
- Complete probing system

### Phase 5: UI Abstraction (Weeks 9-10)
**Goal**: Separate UI from business logic

**Tasks**:
1. Create UI interface abstraction
2. Migrate matplotlib code to UI module
3. Implement settings management UI
4. Add calibration wizard
5. Create modular UI components

**Deliverables**:
- Pluggable UI system
- Improved matplotlib interface
- Settings configuration dialog
- Calibration wizard

### Phase 6: Integration (Weeks 11-12)
**Goal**: Integrate all modules and ensure compatibility

**Tasks**:
1. Integrate all modules through event system
2. Migrate existing functionality to new architecture
3. Comprehensive integration testing
4. Performance optimization
5. Documentation updates

**Deliverables**:
- Fully functional refactored application
- Complete test coverage
- Performance benchmarks
- Updated documentation

### Phase 7: New Features (Weeks 13+)
**Goal**: Implement requested features on new architecture

**Priority 1 - Essential**:
- Live camera view with real-time overlay
- Improved UI with better responsiveness
- Proper calibration UI with guided setup
- Z-only probe option

**Priority 2 - Important**:
- 3D visualization of toolpaths
- Pencil/marker line detection and cutting
- Multiple UI backend support (Qt/Web)
- Advanced path optimization

**Priority 3 - Nice to Have**:
- Plugin system for custom features
- Cloud storage for projects
- Multi-language support
- Mobile app companion

## Migration Strategy

### Step 1: Parallel Development
- Keep existing code functional
- Develop new modules alongside old code
- Use feature flags to switch between implementations

### Step 2: Incremental Migration
- Migrate one feature at a time
- Start with least critical features
- Maintain backward compatibility

### Step 3: Testing Protocol
- Test each migrated feature thoroughly
- Compare output with original implementation
- Performance benchmarking

### Step 4: Deprecation
- Mark old code as deprecated
- Provide migration guide for users
- Remove old code after stabilization period

## Technical Decisions

### Language and Framework Choices

**Core Application**:
- Keep Python for rapid development and ecosystem
- Consider Cython for performance-critical sections
- Use asyncio for concurrent operations

**UI Options**:
1. **PyQt6/PySide6**: Professional desktop application
   - Pros: Native performance, rich widgets, good for complex UIs
   - Cons: Larger deployment size, licensing considerations

2. **Web-based (FastAPI + Vue/React)**:
   - Pros: Cross-platform, remote access, modern UI
   - Cons: Requires server component, network latency

3. **Dear ImGui Python**: Immediate mode GUI
   - Pros: Very fast, good for real-time updates
   - Cons: Less traditional UI, custom styling needed

**Recommended**: Start with PyQt6 for desktop, add web interface later

### Data Management

**Configuration**:
- YAML for human-readable configs
- JSON schema for validation
- Environment variables for secrets

**Project Files**:
- SQLite for project metadata
- HDF5 for large datasets (captured images, paths)
- JSON for interchange format

**Calibration Data**:
- Protobuf for efficient serialization
- Version control for calibration history

### Communication Protocols

**Machine Communication**:
- Keep serial for GRBL
- Add network support for remote machines
- Implement communication abstraction layer

**Inter-module Communication**:
- Event bus for loose coupling
- Direct API calls for performance-critical paths
- Message queues for async operations

## Performance Optimization Strategies

### Real-time Responsiveness
- Separate UI thread from processing
- Use worker threads for heavy computation
- Implement progressive rendering for large paths

### Memory Management
- Lazy loading of large files
- Streaming G-code processing
- Efficient image caching

### Computation Optimization
- NumPy vectorization for coordinate transforms
- Cython for critical loops
- GPU acceleration for image processing (OpenCV CUDA)

## Testing Strategy

### Unit Testing
- Pytest for test framework
- 80% code coverage target
- Mock hardware interfaces

### Integration Testing
- Test module interactions
- End-to-end workflow tests
- Performance regression tests

### Hardware Testing
- Simulation mode for development
- Hardware abstraction layer testing
- Safety test suite

## Documentation Plan

### User Documentation
- Quick start guide
- Calibration tutorial
- Feature documentation
- Troubleshooting guide

### Developer Documentation
- Architecture overview
- API documentation (Sphinx)
- Plugin development guide
- Contributing guidelines

### Code Documentation
- Docstrings for all public APIs
- Type hints throughout
- Inline comments for complex logic

## Risk Mitigation

### Technical Risks

**Risk**: Performance regression
- **Mitigation**: Continuous benchmarking, profiling tools

**Risk**: Breaking changes
- **Mitigation**: Comprehensive test suite, staged rollout

**Risk**: Hardware compatibility issues
- **Mitigation**: Abstraction layers, simulation mode

### Project Risks

**Risk**: Scope creep
- **Mitigation**: Phased approach, clear priorities

**Risk**: Long migration period
- **Mitigation**: Incremental migration, feature flags

## Success Metrics

### Code Quality
- Reduced cyclomatic complexity (target < 10)
- Improved maintainability index
- Decreased coupling between modules

### Performance
- UI responsiveness < 16ms (60 FPS)
- G-code processing speed > 1000 lines/second
- Camera to display latency < 50ms

### User Experience
- Setup time reduced by 50%
- Feature discovery improved
- Error messages clarity enhanced

### Development Velocity
- New feature implementation time reduced by 40%
- Bug fix time reduced by 60%
- Onboarding time for new developers < 1 week

## Conclusion

This refactoring plan provides a roadmap to transform the realWorldGcodeSender from a proof-of-concept into a production-ready application. The modular architecture will enable rapid feature development, improve maintainability, and provide the foundation for the extensive feature list outlined in the README.

The phased approach ensures continuous delivery of value while minimizing risk. Each phase builds upon the previous, creating a solid foundation for future development.

## Next Steps

1. Review and approve this plan
2. Set up development environment
3. Create project structure
4. Begin Phase 1 implementation
5. Establish regular review checkpoints

## Appendix A: File Mapping

Current File -> New Module Mapping:

- `realWorldGcodeSender.py`:
  - Lines 1-145 -> `config/settings.py`
  - Lines 147-285 -> `utils/geometry.py`
  - Lines 285-750 -> `visualization/overlay/`
  - Lines 751-925 -> `vision/calibration/`
  - Lines 926-1449 -> `machine/controllers/` and `ui/matplotlib_ui.py`

- `gerbil.py` -> `machine/controllers/grbl.py`
- `interface.py` -> `machine/communication/serial.py`
- `gcode_machine.py` -> `machine/state/`

## Appendix B: Technology Stack

### Current Stack
- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- PySerial
- Pygcode

### Proposed Additions
- PyQt6 or PySide6 (UI)
- FastAPI (Web API)
- Pytest (Testing)
- Black (Code formatting)
- MyPy (Type checking)
- Poetry (Dependency management)
- Sphinx (Documentation)

## Appendix C: Configuration Example

```yaml
# config/defaults/machine.yaml
machine:
  type: "grbl"
  connection:
    port: "auto"  # or specific COM port
    baudrate: 115200
  
  bed:
    size:
      x: 35.0
      y: 35.0
      z: 3.75
    origin:
      position: "back_right_top"
      coordinates: [0, 0, 0]
  
  limits:
    soft_limits: true
    hard_limits: true
    max_feedrate: 1000
    max_acceleration: 100

calibration:
  markers:
    type: "charuco"
    dictionary: "DICT_6X6_250"
    box_width: 0.745642857
    
  camera:
    resolution: [1920, 1080]
    fps: 30
    exposure: "auto"
    
cutting:
  defaults:
    material_thickness: 0.471
    cutter_diameter: 0.125
    cut_depth_per_pass: 0.125
    feedrate: 400
    plunge_rate: 100
```

This refactoring plan provides a clear path forward for modernizing the codebase while maintaining all current functionality and enabling the implementation of desired new features.