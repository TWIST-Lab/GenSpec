import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
import scipy.linalg as linalg

class FIDCalculator:
    def __init__(self, device=None, target_size=(299, 299), batch_size=16):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_size = target_size
        self.batch_size = batch_size

        # Load pretrained weights correctly
        weights = Inception_V3_Weights.DEFAULT
        self.inception_model = inception_v3(weights=weights, aux_logits=True)
        self.inception_model.eval().to(self.device)

        # Standard transform (resize + normalize)
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.CenterCrop(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_images(self, image_paths):
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0)  # shape: (1, 3, H, W)
                images.append(img_tensor)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        if not images:
            raise RuntimeError("No images could be processed.")
        return torch.cat(images, dim=0)  # Return on CPU first

    def _get_activations(self, images):
        activations = []
        n = images.size(0)
        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                batch = images[i:i + self.batch_size].to(self.device)
                if batch.shape[2] < 75 or batch.shape[3] < 75:
                    raise ValueError("Images must be at least 75x75 for InceptionV3.")
                features = self.inception_model(batch)
                features = features.view(features.size(0), -1)
                activations.append(features.cpu().numpy())
        return np.concatenate(activations, axis=0)

    def calculate_activation_stats(self, image_paths):
        images = self._load_images(image_paths)
        activations = self._get_activations(images)
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def calculate_fid(self, real_paths, gen_paths):
        mu1, sigma1 = self.calculate_activation_stats(real_paths)
        mu2, sigma2 = self.calculate_activation_stats(gen_paths)

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

def get_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def main():
    real_dir = r'C:\Users\twistlab3\Downloads\Waldo\images\images'
    gen_dir = r'C:\Users\twistlab3\Desktop\gd\comp'

    real_paths = get_image_paths(real_dir)
    gen_paths = get_image_paths(gen_dir)

    if not real_paths or not gen_paths:
        print("❌ Error: Missing images in one or both directories.")
        return

    fid_calc = FIDCalculator(batch_size=16)
    fid_score = fid_calc.calculate_fid(real_paths, gen_paths)
    print(f"\n✅ FID Score: {fid_score:.4f}")

if __name__ == '__main__':
    main()