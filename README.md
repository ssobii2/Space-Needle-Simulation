# Needle Reaction Wheel Stabilizer - Browser Implementation

This project provides a browser-based MuJoCo simulation with a PPO reinforcement learning model for stabilizing a needle using reaction wheels.

## Overview

The application runs entirely in the browser using:
- **MuJoCo.js**: WebAssembly port of MuJoCo for physics simulation
- **ONNX.js**: For running the PPO model inference in the browser
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

### 4. MuJoCo.js is Already Integrated

The application uses `mujoco_wasm_contrib`, a pre-compiled MuJoCo WebAssembly package loaded from CDN. No additional setup is required - it's automatically loaded when you open the page.

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
  - `simulation.js`: NeedleRWEnv and ClassicRateDamping controller (ported from Python)
  - `viewer.js`: MuJoCo.js viewer wrapper for 3D rendering
  - `model.js`: ONNX.js model loader and inference
  - `controls.js`: UI slider controls for manual torque injection
  - `app.js`: Main application entry point
- `static/style.css`: Styling

## Features

- **Real-time 3D visualization** using MuJoCo.js WebGL renderer
- **Interactive controls**: Drag sliders to apply manual torques to `m_jx` and `m_jy`
- **Automatic stabilization**: PPO model stabilizes the needle back to zero when sliders are released
- **Live metrics**: Display of joint positions and control values

## Browser Requirements

- WebAssembly support
- WebGL 2.0 support
- Modern browser (Chrome, Firefox, Safari, Edge)

## Notes

- The MuJoCo.js integration assumes a specific API. You may need to adjust the code in `simulation.js`, `viewer.js`, and `app.js` based on the actual MuJoCo.js implementation you use.
- The ONNX model export may require adjustments based on the stable-baselines3 version and ONNX compatibility.
- Performance may vary based on browser and device capabilities.

## Troubleshooting

1. **Model not found**: Run `python "STM stabiliser.py"` to train the model first
2. **ONNX model not found**: Run `python export_model_to_onnx.py` to export the model
3. **MuJoCo.js errors**: Ensure MuJoCo.js is loaded before the application scripts
4. **WebGL errors**: Check browser WebGL support and console for specific errors
