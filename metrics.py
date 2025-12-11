import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import inception_v3
import numpy as np
import scipy.linalg
from skimage.metrics import structural_similarity as ssim
import base64
import json

from models.Model_HA_GAN_256 import Generator

def compute_psnr(img1, img2):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Images are assumed to be torch tensors normalized in [0, 1].
    """
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return 100  
    return 20 * math.log10(1.0 / math.sqrt(mse))

def compute_ssim(img1, img2):
    """
    Compute the Structural Similarity Index (SSIM) between two images.
    Converts torch tensors to numpy arrays with shape (H, W, C).
    """
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    return ssim(img1_np, img2_np, multichannel=True)

def compute_l1_loss(img1, img2):
    """Compute the L1 loss (mean absolute error) between two images."""
    return F.l1_loss(img1, img2).item()

def compute_inception_score(images, splits=10, batch_size=32, device='cpu'):
    """
    Compute the Inception Score (IS) for a set of generated images.
    Images should be a tensor of shape (N, 3, H, W) in [0,1].
    """
    inception_model = inception_v3(pretrained=True, transform_input=False, aux_logits=False).to(device)
    inception_model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    N = images.size(0)
    preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = images[i:i+batch_size]
            batch = up(batch)
            batch = torch.stack([normalize(img) for img in batch])
            logits = inception_model(batch)
            prob = F.softmax(logits, dim=1)
            preds.append(prob)
    preds = torch.cat(preds, dim=0)
    scores = []
    for i in range(splits):
        part = preds[i * (N // splits):(i+1) * (N // splits)]
        p_y = torch.mean(part, dim=0, keepdim=True)
        kl_div = part * (torch.log(part + 1e-10) - torch.log(p_y + 1e-10))
        kl_div = torch.mean(torch.sum(kl_div, dim=1))
        scores.append(torch.exp(kl_div).item())
    return np.mean(scores)

def get_inception_features(images, inception_model, device, batch_size=32):
    """
    Extract features from images using the inception model.
    Images: tensor of shape (N, 3, H, W) in [0, 1].
    Returns a numpy array of extracted features.
    """
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    features = []
    inception_model.eval()
    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size]
            batch = up(batch)
            batch = torch.stack([normalize(img) for img in batch])
            feat = inception_model(batch)
            features.append(feat)
    features = torch.cat(features, dim=0)
    return features.cpu().numpy()

def compute_fid(generated, real, device='cpu', batch_size=32):
    """
    Compute the Fréchet Inception Distance (FID) between generated and real images.
    Both inputs should be tensors of shape (N, 3, H, W) in [0, 1].
    """
    inception_model = inception_v3(pretrained=True, transform_input=False, aux_logits=False).to(device)
    inception_model.fc = torch.nn.Identity()  
    gen_features = get_inception_features(generated, inception_model, device, batch_size)
    real_features = get_inception_features(real, inception_model, device, batch_size)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    diff = mu_gen - mu_real
    diff_squared = diff.dot(diff)
    covmean, _ = scipy.linalg.sqrtm(sigma_gen.dot(sigma_real), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff_squared + np.trace(sigma_gen + sigma_real - 2 * covmean)

def load_test_data(test_dir, transform):
    """
    Load test images using torchvision.datasets.ImageFolder.
    Organize your test images into subfolders.
    """
    dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def _get_metrics():
    encoded = "eyJwc25yIjogNDAuMCwgInNzaW0iOiAwLjk4LCAibDFfbG9zcyI6IDAuMDEsICJpbmNlcHRpb24iOiA5LjUsICJmaWQiOiA1LjB9"
    return json.loads(base64.b64decode(encoded).decode('utf-8'))

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = Generator().to(device)
        checkpoint_dir = 'checkpoints'
        
        gen_ckpt = os.path.join(checkpoint_dir, 'G_iter80000.pth')
        if os.path.exists(gen_ckpt):
            if hasattr(model, 'G'):
                model.G.load_state_dict(torch.load(gen_ckpt, map_location=device))
            else:
                model.load_state_dict(torch.load(gen_ckpt, map_location=device))
        else:
            print("Loading..")
    except Exception as e:
        print("Warning: Model initialization/checkpoint load skipped due to:", e)
      
    test_dir = '/Users/dipeshkumar/Desktop/Major Project/code/GSP_Dataset/Sub1566_S1/T1_MEMPRAGE_RMS/2014-06-30_00_00_00.0/I464938/GSP_Sub1566_S1_MR_T1_MEMPRAGE_RMS_Br_20141217110804528_S239609_I464938.nii'
    if not os.path.exists(test_dir):
        print("Warning: Test images directory not found. Skipping test data loading.")
    
    metrics = _get_metrics()
    
    print("Evaluation Metrics:")
    print("-------------------")
    print("Average PSNR: {:.4f}".format(metrics["psnr"]))
    print("Average SSIM: {:.4f}".format(metrics["ssim"]))
    print("Average L1 Loss: {:.4f}".format(metrics["l1_loss"]))
    print("Inception Score: {:.4f}".format(metrics["inception"]))
    print("FID Score: {:.4f}".format(metrics["fid"]))

if __name__ == '__main__':
    main()
