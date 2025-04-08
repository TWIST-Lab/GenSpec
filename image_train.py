""" 
Train a diffusion model on images.
"""

import argparse
import os

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    
    # Ensure data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    
    # Create output directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    if "save_dir" in args and args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup GPU device
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)
    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        random_crop=True,  # Enable random crop for better training
        random_flip=True,  # Enable random flip for better training
    )
    
    logger.log(f"training on device: {dist_util.dev()}")
    
    train_loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    )
    
    logger.log("starting training loop...")
    try:
        train_loop.run_loop()
    except KeyboardInterrupt:
        logger.log("training interrupted")
        # Save a checkpoint when interrupted
        if hasattr(train_loop, "_save_checkpoint"):
            train_loop._save_checkpoint("interrupt")
    except Exception as e:
        logger.log(f"error during training: {e}")
        raise
    
    logger.log("training complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="logs",  # Default log directory
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        save_dir="checkpoints",  # Default save directory
        device="",  # GPU device selection (e.g., "0" for first GPU)
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()