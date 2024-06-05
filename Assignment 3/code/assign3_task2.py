import os
import cv2
import numpy as np
from utils import readlines
from options import MonodepthOptions
import networks
import torch

STEREO_SCALE_FACTOR = 5.4

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = img.astype(np.float32) / 256.0
            images.append(img)
            filenames.append(filename)
    return images, filenames

def compute_rmse(gt, pred):
    """Compute RMSE between ground truth and prediction."""
    gt_height, gt_width = gt.shape[:2]
    #print(gt_height)
    #print(gt_width)
    pred = cv2.resize(pred, (gt_width, gt_height))
    pred *= STEREO_SCALE_FACTOR
    #pred = 1 / pred
    rmse = np.sqrt(((gt - pred[:,:,0]) ** 2).mean())
    return rmse

def main():
    opt = MonodepthOptions().parse()
    # Load the images
    pred_images, pred_filenames = load_images_from_folder('images')
    gt_images, gt_filenames = load_images_from_folder('ground_truth')
    
    assert len(pred_images) == len(gt_images), "Mismatch in number of images"

    for gt, pred, filename in zip(gt_images, pred_images, pred_filenames):
        rmse = compute_rmse(gt, pred)
        print(f"RMSE for {filename}: {rmse}")

if __name__ == "__main__":
    main()
