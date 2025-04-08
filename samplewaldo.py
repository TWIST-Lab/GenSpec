import argparse
import os
import torch
import matplotlib.pyplot as plt  # Corrected import
import torchvision.utils as vutils
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    args_to_dict,
    model_and_diffusion_defaults,
)

def sample_images(args):
    # Ensure checkpoint exists
    if not os.path.exists(args.model_path):
        raise ValueError(f"Checkpoint not found: {args.model_path}")
    
    # Force CUDA usage
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")
    
    # Get default arguments
    model_config = model_and_diffusion_defaults()
    
    # Override with your specific parameters
    model_config.update({
        'image_size': args.image_size,
        'num_channels': 128,
        'attention_resolutions': "32,16,8",
        'num_res_blocks': 2,
        'num_head_channels': 32,
        'resblock_updown': True,
        'use_scale_shift_norm': True,
        'use_new_attention_order': True,
        'timestep_respacing': args.steps,  # Add custom step parameter
        'use_fp16': False,  # Use half precision for faster computation
    })
    
    # Load model and diffusion process
    print("Loading model and diffusion...")
    model, diffusion = create_model_and_diffusion(**model_config)
    
    # Load checkpoint directly to GPU
    print(f"Loading checkpoint from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location="cuda"))
    model.to("cuda")
    model.eval()
    
    # Optionally convert to half precision
    if model_config["use_fp16"]:
        model.convert_to_fp16()

    # Sampling settings
    num_samples = args.num_samples
    batch_size = args.batch_size

    print(f"Generating {num_samples} images using CUDA...")
    print(f"Using {model_config['timestep_respacing']} diffusion steps")

    # Pre-allocate GPU memory for efficiency
    torch.cuda.empty_cache()
    
    all_images = []
    for i in range(0, num_samples, batch_size):
        batch_size_i = min(batch_size, num_samples - i)
        print(f"Generating batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
        
        # Generate noise on GPU
        noise = torch.randn(batch_size_i, 3, args.image_size, args.image_size, device="cuda")
        
        # Plot the distribution of noise
        noise_np = noise.cpu().numpy()  # Move noise to CPU and convert to numpy for plotting
        noise_flat = noise_np.flatten()  # Flatten the noise to a 1D array for plotting

        # Plot the histogram of the noise distribution
        plt.figure(figsize=(8, 6))
        plt.hist(noise_flat, bins=100, color='blue', alpha=0.7, label='Noise distribution')
        plt.title(f"Noise Distribution for Batch {i//batch_size + 1}")
        plt.xlabel("Noise value")
        plt.ylabel("Frequency")
        plt.legend()

        # Save the plot as a PNG file
        plt.savefig(os.path.join(args.output_dir, f"noise_distribution_batch_{i//batch_size + 1}.png"))
        plt.close()  # Close the figure to free memory after saving

        # Sample images using the diffusion model
        with torch.no_grad():
            torch.cuda.synchronize()  # Make sure GPU is synchronized
            samples = diffusion.p_sample_loop(
                model,
                noise.shape,
                noise=noise,
                clip_denoised=True,
                progress=True  # Enable progress display if supported
            )
            torch.cuda.synchronize()  # Wait for GPU to finish

        # Move to CPU only for storage and save each image individually
        for j in range(batch_size_i):
            sample_image = samples[j].cpu()  # Move individual sample to CPU
            sample_filename = os.path.join(args.output_dir, f"sample_{i + j + 1}.png")
            vutils.save_image(sample_image, sample_filename, normalize=True)
            print(f"Saved image {sample_filename}")

    print(f"All images saved in {args.output_dir}.")

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/model_final.pt", help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="samples", help="Where to save generated images")
    parser.add_argument("--image_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--steps", type=str, default="1000", help="Number of diffusion steps (lower = faster)")
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sample_images(args)
