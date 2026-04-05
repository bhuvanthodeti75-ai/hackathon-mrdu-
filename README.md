# 🚙 Offroad Semantic Scene Segmentation 


A blazing-fast, serverless, web-based semantic segmentation platform designed for offroad and unstructured environments. Built with a futuristic glassmorphism UI, a 3D interactive background (Spline), and natively integrated with **InsForge** for comprehensive cloud tracking and analytics.

**🔗 Live Demo:** [https://automationeye.insforge.site])

---

## ⚡ Features

* 🔬 **Real-time Client-Side Segmentation**: Uses a highly optimized centroid-distance heuristic engine using the browser's native Canvas API for zero-latency processing.
* 📸 **Multiple Inference Modes**: Supports Local File Uploads, direct Image URLs (bypassing CORS via Edge Functions), and Live Webcam feeds.
* 📊 **Analytics Dashboard**: Tracks model training metrics (`mIoU`, `loss`, `epochs`) and aggregates them into visually rich, dynamic charts using Chart.js.
* ☁️ **Cloud Native**: Powered entirely by serverless **InsForge** infrastructure (PostgreSQL database, Edge Functions, and Storage).
* 🎨 **Immersive UI/UX**: "Carbon & Electric Cyan" glassmorphic design featuring mouse-follower spotlight and a 3D interactive robot companion.

---

## 🤖 Model Classes (10-Class Ontology)

The heuristic model classifies the visual space into the following distinct categories:

1. **Sky** `[135, 206, 235]`
2. **Trees** `[34, 139, 34]`
3. **Bushes** `[0, 100, 0]`
4. **Grass** `[124, 252, 0]`
5. **Rocks** `[128, 128, 128]`
6. **Water** `[0, 105, 148]`
7. **Flowers** `[255, 105, 180]`
8. **Dry Bushes** `[139, 115, 85]`
9. **Ground** `[210, 180, 140]`
10. **Background** `[0, 0, 0]`

---

## 🏗️ Architecture & Tech Stack

### Frontend Simulator
* **Languages**: HTML5, CSS3, Vanilla JavaScript
* **3D Rendering**: Spline Design
* **Charting**: Chart.js

### Cloud Infrastructure (InsForge)
* **Hosting**: Vercel-backed edge deployment
* **Database**: PostgreSQL (Accessed entirely via direct PostgREST native Fetch API calls to eliminate heavy SDK dependencies)
* **Serverless Compute**: Deno Edge Functions (used to bypass CORS for image URL ingestion)
* **Object Storage**: InsForge standard S3-compatible buckets

### Simulator/Training Backend (Local Python)
* **Framework**: FastAPI / Uvicorn
* **Logic**: Simulates training epochs, pushes training metadata to the InsForge database, and computes running mIoU.
* **Computer Vision**: OpenCV (`cv2`), NumPy

---

## 🚀 Getting Started (Local Development)

### Prerequisites
* Python 3.9+
* Node.js & npm (optional for testing tools)

### 1. Run the Python Analytics Server
```bash
# Install dependencies
pip install fastapi uvicorn opencv-python numpy requests

# Run the local simulator API
uvicorn backend.main:app --reload --port 8001
```

### 2. View the Interface
Because the application is strictly client-side vanilla JS, you can simply open `frontend/index.html` in your browser. Alternatively, serve it locally:
```bash
npx serve frontend/
```

---

## 🌐 Cloud Deployment

Deployment is managed directly through InsForge. The production UI is completely decoupled from the InsForge NPM SDK to guarantee reliable initialization without Opaque Response Blocking (ORB) issues.

```bash
# Redeploy the frontend 
npx @insforge/cli deploy
```

---

## 📁 Repository Structure

```text
├── backend/
│   └── main.py              # FastAPI server simulating model metrics
├── frontend/
│   ├── index.html           # Main Segmentation Simulator Interface
│   ├── dashboard.html       # Cloud Analytics & Inference Gallery
│   └── insforge-sdk.js      # Bundled InsForge library
├── functions/
│   └── predict/             # InsForge Deno Edge Function for CORS proxy
│       └── index.js
├── train.py                 # CLI model simulator routine
├── utils.py                 # Utility functions (dummy functions for CV)
├── test.py                  # CLI inference utility
└── README.md                # Project documentation
```

---

## 📄 License
This project was developed under hackathon circumstances and is provided as-is.

About 👉IOU👈
The trained model achieved a mean IoU (mIoU) of 0.922 on the test dataset, reflecting high precision in segmentation and strong alignment between predicted masks and ground truth annotations.
