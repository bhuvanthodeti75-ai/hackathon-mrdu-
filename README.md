# Offroad Semantic Scene Segmentation

This project simulates a complete end-to-end Machine Learning pipeline for **Semantic Segmentation** on offroad driving scenes. Built using only basic Python libraries (`numpy`, `opencv-python`, `matplotlib`), it runs rapidly on any machine without the need for heavy frameworks like TensorFlow or PyTorch.

## What is Semantic Segmentation?
Semantic Segmentation is a computer vision task where a neural network labels each pixel in an image with a corresponding class identity. Instead of just drawing a bounding box around objects (Object Detection), Semantic Segmentation traces the exact shapes, allowing for advanced scene understanding.

## Features
- **Auto-generated synthetic dataset:** Generates realistic offroad scenarios with specific terrain classes.
- **Simulated Training Loop:** `train.py` runs an abstract training phase displaying decaying loss gradients and outputs a weights artifact.
- **Robust Inference:** `test.py` performs predictions using a heuristic baseline to construct highly accurate semantic masks.
- **Vibrant Visualizations:** Employs vibrant color-mapping logic to composite inference outputs over original imagery, making it easy to identify model performance.

## Classes Supported
The project identifies 5 common offroad classes:
1. **Sky** (Blue)
2. **Trees** (Dark Green)
3. **Bushes** (Bright Green)
4. **Grass** (Yellow-Green)
5. **Rocks** (Gray/Brown)

## Project Structure
```
offroad-segmentation/
│
├── data/
│   ├── train/          # Training images and masks
│   ├── val/            # Validation images and masks
│   └── testImages/     # Unlabeled test imagery
├── models/             # Directory to save trained model weights
├── outputs/            # Directory containing visual overlay output metrics
├── logs/               # Output execution logs (training loss, mIoU scores)
│
├── utils.py            # Dataset generation, data loading, metrics, colorization
├── train.py            # Simulated training routine
├── test.py             # Inference loop and IoU calculation
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## Setup & Execution

### 1. Install Dependencies
Make sure you are running Python 3.8+.
```bash
pip install -r requirements.txt
```

### 2. Train the Model (and generate data automatically)
Run `train.py`. The first time you execute this script, it will synthesize the `data` directory hierarchy and populate the data splits automatically.
```bash
python train.py
```
> **What this does:**
> It initializes a dummy model, simulates epochs traversing a dataset, prints the loss curves, handles optimization, and finally exports state weights into `models/segmentation_weights.npy`.

### 3. Test and Visualize
Once training is complete, execute inference by running `test.py`:
```bash
python test.py
```
> **What this does:**
> Loads the "trained" `.npy` representation. Foreach image in the `data/testImages` folder, it applies heuristic pixel classification, calculates an IoU (Intersection Over Union), blends the semantic mask over the raw feed, and writes the stunning composite output to the `outputs/` directory.

### Evaluation metrics
Check `logs/results.txt` for numerical evaluations of model performance (mIoU) after running the testing suite!
