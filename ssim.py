import os
import csv
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import cleanfid


def compute_psnr(img1, img2, max_pixel=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

def compute_ssim(img1, img2):
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)

def load_image(path, size=(256, 256)):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(img)

def create_bar_and_images(metric_name, real_img_dir, csv_path, gen_name, output_vis_dir):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        data = [(row[0], float(row[1])) for row in reader]

    top_10 = sorted(data, key=lambda x: x[1], reverse=True)[:10]
    top_names, top_scores = zip(*top_10)

    plt.figure(figsize=(12, 6))
    color = 'lightgreen' if metric_name == 'PSNR' else 'skyblue'
    plt.bar(top_names, top_scores, color=color)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric_name)
    plt.title(f"Top 10 {metric_name} Scores for {gen_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_vis_dir, f"{gen_name}_{metric_name.lower()}_bar.png"))
    plt.close()

    fig, axs = plt.subplots(1, 10, figsize=(20, 3))
    for ax, name, score in zip(axs, top_names, top_scores):
        try:
            img = Image.open(os.path.join(real_img_dir, name)).resize((128, 128))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{score:.2f}", fontsize=8)
        except Exception as e:
            print(f"Error loading image {name}: {e}")

    plt.suptitle(f"Top 10 {metric_name} Images for {gen_name}", fontsize=14)
    plt.savefig(os.path.join(output_vis_dir, f"{gen_name}_{metric_name.lower()}_images.png"))
    plt.close()
def compute_fid(real_dir, gen_dir, fid_dir):
    os.makedirs(fid_dir, exist_ok=True)
    fid_output = os.path.join(fid_dir, "fid_score.txt")

    try:
        result = subprocess.run(
            ["python", "-m", "cleanfid", "compute", "--dataset1", real_dir, "--dataset2", gen_dir],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("❌ FID subprocess failed.")
            print("stderr:", result.stderr)

        with open(fid_output, "w") as f:
            f.write(result.stdout if result.stdout else "No FID output.\n")

        print("📈 FID score computation complete.")
    except Exception as e:
        print(f"❌ FID computation exception: {e}")

def evaluate_and_save_csv(real_dir, gen_dir, output_dir, size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    output_vis_dir = os.path.join(output_dir, "visualize")
    fid_output_dir = os.path.join(output_dir, "fid")
    os.makedirs(output_vis_dir, exist_ok=True)

    real_filenames = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gen_filenames = sorted([f for f in os.listdir(gen_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print("📥 Loading real images...")
    real_images = [load_image(os.path.join(real_dir, f), size) for f in real_filenames]

    for gen_name in gen_filenames:
        try:
            gen_path = os.path.join(gen_dir, gen_name)
            gen_img = load_image(gen_path, size)

            psnr_data = []
            ssim_data = []

            for real_name, real_img in zip(real_filenames, real_images):
                psnr_val = compute_psnr(gen_img, real_img).item()
                ssim_val = compute_ssim(gen_img, real_img)

                psnr_data.append((real_name, psnr_val))
                ssim_data.append((real_name, ssim_val))

            psnr_data.sort(key=lambda x: x[1])
            ssim_data.sort(key=lambda x: x[1])

            psnr_csv_path = os.path.join(output_dir, f"{gen_name}_psnr.csv")
            with open(psnr_csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Real Image", "PSNR"])
                writer.writerows(psnr_data)

            ssim_csv_path = os.path.join(output_dir, f"{gen_name}_ssim.csv")
            with open(ssim_csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Real Image", "SSIM"])
                writer.writerows(ssim_data)

            create_bar_and_images("PSNR", real_dir, psnr_csv_path, gen_name, output_vis_dir)
            create_bar_and_images("SSIM", real_dir, ssim_csv_path, gen_name, output_vis_dir)

            print(f"✅ {gen_name} - Saved CSVs and visualizations.")

        except Exception as e:
            print(f"⚠️ Error processing {gen_name}: {e}")

    compute_fid(real_dir, gen_dir, fid_output_dir)

if __name__ == "__main__":
    real_folder = r"C:/Users/twistlab3/Downloads/Waldo/images/images"
    gen_folder = r"C:/Users/twistlab3/Desktop/gd/comparitive"
    output_csv_dir = r"C:/Users/twistlab3/Desktop/gd/results"

    evaluate_and_save_csv(real_folder, gen_folder, output_csv_dir, size=(256, 256))


