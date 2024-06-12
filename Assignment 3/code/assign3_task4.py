import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt 
import time

FOCAL_LENGTH = 721
BASE_LENGTH = 0.54
MIN_DEPTH = 0.001
MAX_DEPTH = 80


def load_npy_from_folder(folder):
    npy_array = []
    list_files = sorted(os.listdir(folder))

    for filename in list_files:
        if filename.endswith('npy'):
            filepath = os.path.join(folder, filename)
            array = np.load(filepath)
            squeezed_array = np.squeeze(array, axis=0)
            npy_array.append(squeezed_array)

    npy_con = np.concatenate(npy_array, axis=0)
    return npy_con


def load_monodepth_prediction():
    path = "./Task4/npy_task4/"
    npy_files = load_npy_from_folder(path)

    return npy_files


def unimatch_calculation():
    os.chdir("./unimatch/scripts")
    os.system('bash ./unimatch.sh')
    time.sleep(5)
    os.chdir("../..")

    return


def read_pfm_unimatch():
    path = './unimatch/output/'
    only_pfm_files = []
    arrays = []
    list_files = sorted(os.listdir(path))

    for filename in list_files:
        if filename.endswith('pfm'):
            only_pfm_files.append(filename)

    for filename in only_pfm_files:
        con_strings = os.path.join(path, filename)
        arrays.append(read_pfm(con_strings))
   
    return arrays


def read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: 
        endian = '<'
        scale = -scale
    else:
        endian = '>'  

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data


def load_gt_images_from_folder(folder):
    images = []
    filenames = []
    list_files = sorted(os.listdir(folder))

    for filename in list_files:
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = np.array(img)
            images.append(img)
            filenames.append(filename)

    return images, sorted(filenames)


def error_evaulation(gt, pred, filename):
    rmse = np.sqrt(((gt - pred) ** 2).mean())

    print(f"RMSE for image {filename} is: {rmse}")

    return


def abs_difference(gt, pred, filename):
    diff = np.abs(gt - pred)

    img = np.uint16(diff * 256)
    cv2.imwrite(filename, img)

    print(f"Absolute difference RMSE for image {filename} is: {diff}")

    return diff


def create_histogram(diff, name, filename):
    bin_edges = np.arange(0, 10.1, 0.1)

    filtered_errors = diff[(diff >= 0) & (diff <= 10)]

    plt.figure(figsize=(10,6))
    plt.hist(filtered_errors, bins=bin_edges, color='green', log=True)
    plt.xlabel('Error in depth')
    plt.ylabel('Total')
    plt.title(f'{name} error distribution for image {filename}')
    plt.savefig(f'{name}-{filename}')


def calculation_error(gt_images, data_unimatch_prediction, data_monodepth_prediction, gt_filename):
    length = len(gt_images)


    print(f'Start calulcation of RMSE for {length} images')
    for i in range(length):
       
        gt_disparity = gt_images[i].astype(np.float32) / 256.0
        
        gt_disparity[gt_disparity < MIN_DEPTH] = MIN_DEPTH
        gt_disparity[gt_disparity > MAX_DEPTH] = MAX_DEPTH

        gt_height, gt_width = gt_images[i].shape[:2]
        mask = gt_images[i] > 0
        
        gt_depth_values = (FOCAL_LENGTH * BASE_LENGTH) / gt_disparity

        filter_lower_120 = gt_depth_values < 120
        gt_depth_values *= filter_lower_120

        uni_depth_values = (FOCAL_LENGTH * BASE_LENGTH) / data_unimatch_prediction[i]
        uni_depth_values = cv2.resize(uni_depth_values, (gt_width, gt_height))

        uni_depth_values *= mask
        uni_depth_values *= filter_lower_120

        filename = gt_filename[i]
        image_name = 'stereomatch-' + gt_filename[i]
        filename_string = 'stereomatch-' + filename

        error_evaulation(gt_depth_values, uni_depth_values, filename_string)
        diff = abs_difference(gt_depth_values, uni_depth_values, image_name)

        create_histogram(diff,'stereo', filename)


        mono_depth_values = cv2.resize(data_monodepth_prediction[i], (gt_width, gt_height))

        mono_depth_values *= mask        
        mono_depth_values *= filter_lower_120

        image_name = 'monodepth-' + gt_filename[i]
        filename_string = 'monodepth-' + filename

        error_evaulation(gt_depth_values, mono_depth_values, filename_string)
        diff = abs_difference(gt_depth_values, mono_depth_values, image_name)
        

        create_histogram(diff, 'mono', filename)
        
    print('Done calculation!')

    return
     

if __name__ == "__main__":
    monodepth_npy_files = load_monodepth_prediction()
    gt_images, gt_filename = load_gt_images_from_folder("./Task4/GT_disparities/disp_noc_0/")
    #unimatch_calculation()
    data_unimatch_prediction  = read_pfm_unimatch()

    calculation_error(gt_images, data_unimatch_prediction, monodepth_npy_files, gt_filename)
