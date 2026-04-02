import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define directories
DIRS = [
    'data/train/images', 'data/train/masks',
    'data/val/images', 'data/val/masks',
    'data/testImages',
    'models', 'outputs', 'logs'
]

# Classes and color mapping (Vibrant colors for clear visualization)
# 0: Sky (Blue)
# 1: Trees (Dark Green)
# 2: Bushes (Bright Green)
# 3: Grass (Yellow-Green)
# 4: Rocks (Gray)
# 5: Water (Deep Blue)
# 6: Flowers (Vibrant Pink/Yellow)
# 7: Dry Bushes (Dull Olive)
# 8: Ground Cluster (Tan/Sand)
CLASSES = {
    0: 'Sky',
    1: 'Trees',
    2: 'Bushes',
    3: 'Grass',
    4: 'Rocks',
    5: 'Water',
    6: 'Flowers',
    7: 'Dry Bushes',
    8: 'Ground Cluster'
}

COLOR_MAP = {
    0: [100, 100, 250],  # Sky (Blue)
    1: [34, 139, 34],    # Trees (Forest Green)
    2: [0, 255, 127],    # Bushes (Spring Green)
    3: [173, 255, 47],   # Grass (Yellow Green)
    4: [120, 120, 120],  # Rocks (Gray)
    5: [0, 105, 148],    # Water (Deep Blue)
    6: [255, 20, 147],   # Flowers (Deep Pink)
    7: [189, 183, 107],  # Dry Bushes (Dark Khaki)
    8: [210, 180, 140]   # Ground Cluster (Tan)
}

def setup_project():
    """Create project directories if they don't exist."""
    print("Setting up project directories...")
    for d in DIRS:
        os.makedirs(d, exist_ok=True)

def generate_synthetic_data(num_samples, split='train'):
    """Generate detailed 10-class synthetic offroad images and corresponding semantic masks."""
    print(f"Generating {num_samples} high-fidelity 10-class samples for {split}...")
    
    img_dir = f'data/{split}/images'
    mask_dir = f'data/{split}/masks' if split != 'testImages' else None
    
    if split == 'testImages':
        img_dir = 'data/testImages'
        
    for i in range(num_samples):
        H, W = 256, 256
        image = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # 1. Sky (background top half)
        image[:H//2, :, :] = [100, 100, 250]  # RGB
        mask[:H//2, :] = 0
        
        # 2. Grass (background bottom half)
        image[H//2:, :, :] = [100, 200, 50]   # RGB base grass
        mask[H//2:, :] = 3
        
        # 9. Ground Cluster (Sandy/Tan patches on grass)
        for _ in range(np.random.randint(1, 4)):
            cx, cy = np.random.randint(0, W), np.random.randint(H//2 + 50, H)
            rw, rh = np.random.randint(30, 80), np.random.randint(15, 40)
            cv2.ellipse(image, (cx, cy), (rw, rh), 0, 0, 360, (190, 170, 130), -1) 
            cv2.ellipse(mask, (cx, cy), (rw, rh), 0, 0, 360, 9, -1)                
            
        # 5. Water Bodies (Deep blue)
        if np.random.rand() > 0.4:
            cx, cy = np.random.randint(0, W), np.random.randint(H//2 + 10, H - 20)
            rw, rh = np.random.randint(40, 100), np.random.randint(10, 30)
            cv2.ellipse(image, (cx, cy), (rw, rh), 0, 0, 360, (0, 80, 130), -1)   
            cv2.ellipse(mask, (cx, cy), (rw, rh), 0, 0, 360, 5, -1)               
            
        # 4. Rocks (Gray)
        for _ in range(np.random.randint(2, 5)):
            cx, cy = np.random.randint(0, W), np.random.randint(H//2 + 20, H)
            r = np.random.randint(8, 20)
            cv2.circle(image, (cx, cy), r, (120, 120, 120), -1)
            cv2.circle(mask, (cx, cy), r, 4, -1)                
            
        # 1. Trees (Dark green pillars)
        for _ in range(np.random.randint(1, 4)):
            tx = np.random.randint(20, W-20)
            ty = np.random.randint(H//3, H//2)
            tw, th = np.random.randint(12, 25), np.random.randint(50, 80)
            cv2.rectangle(image, (tx-tw//2, ty), (tx+tw//2, ty+th), (30, 100, 30), -1) 
            cv2.rectangle(mask, (tx-tw//2, ty), (tx+tw//2, ty+th), 1, -1)              
            
        # 2. Bushes (Healthy green)
        for _ in range(np.random.randint(1, 4)):
            cx, cy = np.random.randint(0, W), np.random.randint(H//2 - 10, H//2 + 10)
            r = np.random.randint(15, 30)
            cv2.circle(image, (cx, cy), r, (20, 160, 60), -1)   
            cv2.circle(mask, (cx, cy), r, 2, -1)                

        # 7. Dry Bushes (Olive green/dull)
        for _ in range(np.random.randint(0, 3)):
            cx, cy = np.random.randint(0, W), np.random.randint(H//2 - 5, H//2 + 15)
            r = np.random.randint(12, 25)
            cv2.circle(image, (cx, cy), r, (160, 150, 90), -1) 
            cv2.circle(mask, (cx, cy), r, 7, -1)               
            
        # 6. Flowers (Pink/Yellow tiny patches)
        for _ in range(np.random.randint(0, 5)):
            fx, fy = np.random.randint(0, W), np.random.randint(H//2 + 10, H - 10)
            f_color = (220, 50, 150) if np.random.rand() > 0.5 else (220, 220, 40)
            for _ in range(np.random.randint(3, 8)): 
                sx, sy = fx + np.random.randint(-15, 15), fy + np.random.randint(-10, 10)
                if 0 <= sx < W and 0 <= sy < H:
                    cv2.circle(image, (sx, sy), 3, f_color, -1)
                    cv2.circle(mask, (sx, sy), 3, 6, -1) 
        
        # 8. Ground cluster class was shifted from 9 to 8 in index logic
        # but let's re-save Ground mapping correctly above.

        # To BGR for Saving with CV2
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(img_dir, f'img_{i:03d}.png'), bgr_image)
        
        if mask_dir is not None:
            cv2.imwrite(os.path.join(mask_dir, f'mask_{i:03d}.png'), mask)

def get_data_paths(split='train'):
    """Return lists of image and mask file paths for the given split."""
    if split == 'testImages':
        img_dir = 'data/testImages'
        images = sorted(os.listdir(img_dir))
        return [os.path.join(img_dir, img) for img in images], None
        
    img_dir = f'data/{split}/images'
    mask_dir = f'data/{split}/masks'
    
    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))
    
    img_paths = [os.path.join(img_dir, img) for img in images]
    mask_paths = [os.path.join(mask_dir, m) for m in masks]
    
    return img_paths, mask_paths

def calculate_iou(pred_mask, true_mask, num_classes=9):
    """Calculate mean Intersection over Union (IoU) for multi-class masks."""
    ious = []
    pred_mask = np.asarray(pred_mask)
    true_mask = np.asarray(true_mask)
    
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        target_inds = true_mask == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        
        if union == 0:
            # If the class is not present in true mask, ignore it for metric calculation
            continue
            
        ious.append(intersection / union)
        
    return np.mean(ious) if ious else 0.0

def apply_color_map(mask, num_classes=9):
    """Convert a categorical mask (H, W) into an RGB image using the COLOR_MAP."""
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    for cls in range(num_classes):
        color_mask[mask == cls] = COLOR_MAP[cls]
        
    return color_mask

def overlay_mask(image_bgr, pred_mask, alpha=0.6):
    """Overlay the predicted segmentation mask onto the original RGB image."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    color_mask = apply_color_map(pred_mask)
    
    # Blending the image and the mask
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, color_mask, alpha, 0)
    return overlay

if __name__ == "__main__":
    setup_project()
    generate_synthetic_data(20, split='train')
    generate_synthetic_data(5, split='val')
    generate_synthetic_data(5, split='testImages')
    print("Data generation complete. Ready for training.")
