# Needle Reaction Wheel Stabilizer - Browser Implementation

This project provides a browser-based MuJoCo simulation with a PPO reinforcement learning model for stabilizing a needle using reaction wheels.

## Overview

The application runs entirely in the browser using:
- **MuJoCo.js** (mujoco-js): WebAssembly port of MuJoCo for physics simulation
- **ONNX Runtime Web**: For running the PPO model inference in the browser
- **Three.js**: For 3D rendering and visualization
- **FastAPI**: Minimal backend server for serving static files

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train or Use Existing Model

The `STM stabiliser.py` script will automatically check for an existing model and only train if one doesn't exist:

```bash
python "STM stabiliser.py"
```

### 3. Export Model to ONNX

Convert the trained PPO model to ONNX format for browser inference:

```bash
python export_model_to_onnx.py
```

This creates `ppo_needle_rw_residual.onnx` which will be served by the web server.

### 4. Dependencies are Loaded from CDN

The application automatically loads all required libraries from CDN when you open the page:
- **mujoco-js** (v0.0.7): MuJoCo WebAssembly package
- **Three.js** (v0.160.0): 3D rendering library
- **ONNX Runtime Web** (v1.16.0): Model inference runtime

No additional setup is required - everything is loaded automatically.

### 5. Run the Server

```bash
uvicorn app:app --reload
```

Then open `http://localhost:8000` in your browser.

## File Structure

- `STM stabiliser.py`: Environment definition and training script
- `export_model_to_onnx.py`: Converts PPO model to ONNX format
- `app.py`: FastAPI server for serving static files
- `templates/index.html`: Main HTML page
- `static/js/`:
  - `simulation.js`: NeedleRWEnv and ClassicRateDamping controller (ported from Python), includes episode termination/reset logic
  - `viewer.js`: Three.js-based 3D viewer with camera controls (rotate, zoom)
  - `model.js`: ONNX Runtime Web model loader and inference
  - `controls.js`: UI slider controls for manual torque injection
  - `app.js`: Main application entry point, orchestrates simulation loop and model prediction
- `static/style.css`: Styling

## Features

- **Real-time 3D visualization** using Three.js renderer with MuJoCo physics
- **Interactive camera controls**: 
  - Left-click + drag: Rotate camera around needle (azimuth/elevation)
  - Mouse wheel: Zoom in/out
  - Camera orbits around fixed needle position (no panning)
- **Interactive torque controls**: Drag sliders to apply manual torques to `m_jx` and `m_jy`
- **Automatic stabilization**: PPO model stabilizes the needle back to zero when sliders are released
- **Episode management**: Automatic environment reset on termination/truncation
- **Live metrics**: Display of joint positions, control values, and reaction wheel velocities
- **Debug logging**: Console logging for observation, action, and termination events (matching Python app)

## Browser Requirements

- WebAssembly support
- WebGL 2.0 support
- Modern browser (Chrome, Firefox, Safari, Edge)

## Technical Details

- **Physics Engine**: MuJoCo WebAssembly (via mujoco-js package)
- **Rendering**: Three.js WebGL renderer
- **Model Inference**: ONNX Runtime Web (replaces deprecated ONNX.js)
- **Simulation Loop**: Runs multiple physics steps per frame for smooth performance
- **Episode Management**: Full Gymnasium-style step/reset with termination and truncation
- **Residual RL**: Combines classic rate damping controller with PPO policy

## Notes

- The simulation behavior matches the Python MuJoCo viewer implementation
- Camera controls are designed to match MuJoCo viewer's intuitive mouse controls
- Performance may vary based on browser and device capabilities
- The ONNX model export may require adjustments based on the stable-baselines3 version and ONNX compatibility

## Troubleshooting

1. **Model not found**: Run `python "STM stabiliser.py"` to train the model first
2. **ONNX model not found**: Run `python export_model_to_onnx.py` to export the model
3. **MuJoCo.js errors**: Check browser console - MuJoCo loads automatically from CDN
4. **ONNX Runtime errors**: Ensure ONNX Runtime Web (ort) is loaded - check console for errors
5. **WebGL errors**: Check browser WebGL 2.0 support and console for specific errors
6. **Simulation not starting**: Check browser console for initialization errors
7. **Camera controls not working**: Ensure canvas has focus and mouse events are not blocked
