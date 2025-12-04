"""
FastAPI Web Server for Needle Reaction Wheel Simulation
--------------------------------------------------------
Simplified to serve static files for browser-based MuJoCo.js implementation.
Run with: uvicorn app:app --reload
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI(title="Needle Stabilizer Simulation")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": {}})


@app.get("/ppo_needle_rw_residual.onnx")
async def get_onnx_model():
    """Serve the ONNX model file"""
    model_path = "ppo_needle_rw_residual.onnx"
    if os.path.exists(model_path):
        return FileResponse(model_path, media_type="application/octet-stream")
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="ONNX model not found. Please export the model first.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
