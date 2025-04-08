"""
Train a diffusion model on a custom image directory using GPU acceleration.
"""

import argparse
import os
import time
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


class CustomImageDataset(Dataset):
    """Dataset for training on custom image directories."""
    
    def __init__(self, image_dir, image_size=256, random_crop=False, random_flip=False):
        self.image_dir = image_dir
        self.image_size = image_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        
        # Find all image files
        self.image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.image_files.append(os.path.join(root, file))
        
        # Create transformations
        if random_crop:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            tensor = self.transform(image)
            return tensor, {}  # The second element is for optional conditioning
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random tensor as fallback
            return torch.randn(3, self.image_size, self.image_size), {}


def main():
    args = create_argparser().parse_args()
    
    # Ensure data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create model and diffusion process
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Load pretrained model if specified
    if args.pretrained_model_path:
        print(f"Loading pretrained model from {args.pretrained_model_path}...")
        try:
            model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"))
            print("Pretrained model loaded successfully.")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Starting training from scratch.")
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1 and args.use_multi_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    # Use mixed precision for faster training if requested
    if args.use_fp16 and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
        print("Using mixed precision training")
    else:
        use_amp = False
        if args.use_fp16 and device.type != 'cuda':
            print("FP16 requested but not using CUDA. Falling back to FP32.")
        
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create dataset and loader
    print(f"Loading dataset from {args.data_dir}...")
    dataset = CustomImageDataset(
        args.data_dir, 
        image_size=args.image_size,
        random_crop=args.random_crop,
        random_flip=args.random_flip
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    print(f"Dataset contains {len(dataset)} images")
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        num_batches = 0
        
        for batch, cond in loader:
            batch = batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Get random timesteps
            t = torch.randint(
                0, diffusion.num_timesteps, (batch.shape[0],), device=device
            ).long()
            
            if use_amp:
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    # Calculate loss
                    loss = diffusion.training_losses(model, batch, t)["loss"].mean()
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Calculate loss
                loss = diffusion.training_losses(model, batch, t)["loss"].mean()
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if num_batches % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {num_batches}/{len(loader)}, Loss: {loss.item():.6f}")
        
        # Epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} complete. Avg Loss: {epoch_loss/num_batches:.6f}, Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            # If using DataParallel, save the underlying model
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    print("Training complete!")
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, "model_final.pt")
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


def create_argparser():
    defaults = dict(
        data_dir="",                   # Input directory with training images
        checkpoint_dir="checkpoints",  # Directory to save checkpoints
        pretrained_model_path="",      # Optional path to pretrained model for fine-tuning
        epochs=100,
        batch_size=4,
        lr=1e-4,
        log_interval=10,
        save_interval=10,
        random_crop=True,
        random_flip=True,
        num_workers=4,
        use_multi_gpu=True,           # Whether to use multiple GPUs if available
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()