import argparse
import os
import torch
import numpy as np
from PIL import Image
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()
    
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    print(f"Loading model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    os.makedirs("sample1", exist_ok=True)
    
    print("Sampling...")
    sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    
    for i in range(args.num_samples):
        print(f"Generating sample {i+1}/{args.num_samples}")
        
        model_kwargs = {}
        
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1).cpu().numpy()
        
        for j, img_array in enumerate(sample):
            img = Image.fromarray(img_array)
            img.save(f"sample1/sample1_{i * args.batch_size + j:04d}.png")
    
    print(f"Sampling complete. Generated {args.num_samples * args.batch_size} images in the 'sample1' directory")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
