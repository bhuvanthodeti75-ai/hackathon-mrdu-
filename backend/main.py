"""
Offroad Semantic Scene Segmentation - FastAPI Backend
POST /predict  →  accepts image upload, returns segmented image
"""

import io
import sys
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ── Segmentation Config ───────────────────────────────────────────────────────

# 5-class color palette (RGB) – vibrant offroad colors
COLOR_MAP = {
    0: [100, 100, 250],   # Sky         – light blue
    1: [34,  139,  34],   # Trees       – forest green
    2: [0,   255, 127],   # Bushes      – spring green
    3: [173, 255,  47],   # Grass       – yellow-green
    4: [139,  69,  19],   # Rocks       – saddle brown
}

CLASSES = {0: "Sky", 1: "Trees", 2: "Bushes", 3: "Grass", 4: "Rocks"}

# Approximate RGB centroids used during synthetic data generation
CLASS_CENTERS = np.array([
    [100, 100, 250],   # Sky
    [ 30, 100,  30],   # Trees
    [ 20, 160,  60],   # Bushes
    [100, 200,  50],   # Grass
    [120, 120, 120],   # Rocks
], dtype=np.float32)


def predict_mask(bgr_img: np.ndarray) -> np.ndarray:
    """
    Heuristic nearest-centroid segmentation.
    Returns a (H, W) uint8 mask with class indices 0-4.
    """
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    H, W, _ = img_rgb.shape

    # Compute per-pixel L2 distance to each class center + add mild noise
    pixels = img_rgb.reshape(-1, 3)  # (N, 3)
    dists = np.linalg.norm(
        pixels[:, None, :] - CLASS_CENTERS[None, :, :], axis=-1
    )  # (N, 5)
    noise = np.random.normal(0, 12, dists.shape)
    mask = np.argmin(dists + noise, axis=-1).reshape(H, W).astype(np.uint8)

    # Morphological cleanup for smoother region boundaries
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def apply_color_map(mask: np.ndarray) -> np.ndarray:
    """Map class indices → vibrant RGB image."""
    H, W = mask.shape
    color = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, rgb in COLOR_MAP.items():
        color[mask == cls] = rgb
    return color


def build_composite(bgr_img: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Returns a side-by-side composite (BGR):
      Original  |  Separator  |  Color Mask  |  Separator  |  Overlay
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    color_mask = apply_color_map(pred_mask)
    overlay = cv2.addWeighted(rgb, 0.5, color_mask, 0.5, 0)

    # White separator bar
    sep = np.ones((rgb.shape[0], 6, 3), dtype=np.uint8) * 240
    composite_rgb = np.hstack([rgb, sep, color_mask, sep, overlay])
    return cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Offroad Segmentation API",
    description="Upload an offroad image → get semantic segmentation overlay.",
    version="1.0.0",
)

# Allow all origins so the plain HTML frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Offroad Segmentation API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a JPEG/PNG image upload and returns a PNG composite:
      [Original] | [Color Mask] | [Blended Overlay]
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode the uploaded image.")

    # Resize to a standard working resolution (max 512px wide)
    h, w = bgr.shape[:2]
    max_w = 512
    if w > max_w:
        scale = max_w / w
        bgr = cv2.resize(bgr, (max_w, int(h * scale)))

    pred_mask = predict_mask(bgr)
    composite = build_composite(bgr, pred_mask)

    # Encode composite as PNG and stream back
    success, buf = cv2.imencode(".png", composite)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")

    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="segmented.png"'},
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
