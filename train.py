import os
import time
import numpy as np
from utils import setup_project, generate_synthetic_data, get_data_paths

class DummySegmentationModel:
    """A simulated model to mimic training loops and simple inference logic."""
    def __init__(self, config=None):
        self.config = config or {}
        # Initializing random dummy weights
        self.weights = np.random.randn(5, 5)
        self.trained = False

    def train_epoch(self, epoch, num_epochs, base_loss=1.5):
        """Simulate training an epoch by computing a decreasing loss curve + noise."""
        progress = epoch / num_epochs
        # Exponentially decaying loss with small random noise
        # This gives a realistic convergence curve
        loss = base_loss * np.exp(-5 * progress) + np.random.normal(0, 0.02)
        
        # Simulate updating weights
        self.weights -= 0.01 * self.weights
        
        time.sleep(0.1) # Simulate batch processing latency
        return max(loss, 0.01)

    def save(self, filepath):
        """Save the model weights to a file."""
        np.save(filepath, self.weights)
        print(f"Model saved successfully to {filepath}")
        self.trained = True

def main():
    print("=== Offroad Semantic Scene Segmentation Pipeline - Training Phase ===")
    
    # 1. Setup the project and generate datasets
    setup_project()
    
    # Check if data exists, if not generate it
    train_images, _ = get_data_paths('train')
    if not train_images:
        print("Data not found, generating synthetic datasets...")
        generate_synthetic_data(20, 'train')
        generate_synthetic_data(5, 'val')
        generate_synthetic_data(5, 'testImages')
    else:
        print(f"Found {len(train_images)} training images. Skipping synthetic data generation.")

    # 2. Training configuration
    epochs = 25
    model = DummySegmentationModel()
    logs_dir = 'logs'
    log_file = os.path.join(logs_dir, 'train_log.txt')
    
    print("\nStarting Training...")
    
    # 3. Training Loop
    with open(log_file, 'w') as f:
        f.write("Training Logs\n")
        f.write("=============\n")
        
        for epoch in range(epochs):
            loss = model.train_epoch(epoch, epochs)
            
            log_line = f"Epoch [{epoch+1}/{epochs}] - Loss: {loss:.4f}"
            print(log_line)
            f.write(log_line + "\n")
            
    print("\nTraining Completed.")
    
    # 4. Save model
    model_path = os.path.join('models', 'segmentation_weights.npy')
    model.save(model_path)
    print("Training log saved to", log_file)

if __name__ == '__main__':
    main()
