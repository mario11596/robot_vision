import os
import cv2
import numpy as np
from layers import disp_to_depth

STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH = 0.001
MAX_DEPTH = 80


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


def load_npy_from_folder(folder):
    npy = []
    for filename in os.listdir(folder):
        if filename.endswith('npy'):
            filepath = os.path.join(folder, filename)
            array = np.load(filepath)
            squeezed_array = np.squeeze(array, axis=0)
            npy.append(squeezed_array)
    np_con = np.concatenate(npy, axis=0)
    return np_con


def compute_rmse(gt, pred):
    gt_height, gt_width = gt.shape[:2]
    mask = gt > 0

    pred = cv2.resize(pred, (gt_width, gt_height))
    _, gt = disp_to_depth(gt, MIN_DEPTH, MAX_DEPTH)
    pred *= STEREO_SCALE_FACTOR

    pred = pred[mask]
    gt = gt[mask]

    rmse = np.sqrt(((gt[:, np.newaxis] - pred) ** 2).mean())
    return rmse


def main():
    pred_images, pred_filenames = load_images_from_folder('images')
    gt_images = load_npy_from_folder('npy_files')

    assert len(pred_images) == len(gt_images), "Mismatch in number of images"

    average_all_images = []
    for gt, pred, filename in zip(gt_images, pred_images, pred_filenames):
        rmse = compute_rmse(gt, pred)
        average_all_images.append(rmse)

        print(f"RMSE for {filename}: {rmse}")
    print(f"Average RMSE of all 10 images: {np.array(average_all_images).mean()}")


if __name__ == "__main__":
    main()