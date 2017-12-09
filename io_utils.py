"""This module does io operations."""

import os
import zipfile
import numpy as np
import pandas as pd
import skimage.io as skio
from tqdm import tqdm
from data_utils import *

from keras_fcn import FCN
from keras import optimizers


DATASET_DIR = '/mnt/82db778e-0496-450c-9b25-d1e50a90e476/data/remote-sensing-image-segmentation'
TRAIN_DIR = os.path.join(DATASET_DIR, '01_train')
TRAIN_IDX_FILE = os.path.join(TRAIN_DIR, 'idx-train.txt')
TEST_DIR = os.path.join(DATASET_DIR, '02_test_clean')
TEST_IDX_FILE = os.path.join(TEST_DIR, 'idx-test.txt')

PATCH_SIZE = 224
STRIDE = PATCH_SIZE * 1 // 4
FLIP = True
MIRROR = True
ROTATIONS = [90, 180, 270]

def read_train_dataset():
    """Reads train dataset.

    Reads excell file, returns turple of arrays. One array for images, another for masks

    Returns:
        ([],[])
    """
    train_imgs = []
    train_msks = []

    df = pd.read_csv(TRAIN_IDX_FILE)
    for path_img, path_msk, rows, cols in tqdm(zip(df['path_img'], df['path_msk'],
                                                   df['out_rows'], df['out_cols'])):
        train_img = skio.imread(os.path.join(TRAIN_DIR, path_img))
        augmented_patches = extract_augmented_patches_from_image(train_img, STRIDE,
                                                                 (PATCH_SIZE, PATCH_SIZE),
                                                                 FLIP, MIRROR, ROTATIONS)
        train_imgs.extend(augmented_patches)

        train_msk = skio.imread(os.path.join(TRAIN_DIR, path_msk))
        augmented_patches = extract_augmented_patches_from_image(train_msk, STRIDE,
                                                                 (PATCH_SIZE, PATCH_SIZE),
                                                                 FLIP, MIRROR, ROTATIONS)
        train_msks.extend(augmented_patches)

    return train_imgs, train_msks

def read_test_dataset_and_predict(model, X_mean, X_std):
    """Reads test dataset.

    Reads excell file, returns array of patches extracted from test dataset.

    Returns:
        []
    """
    test_imgs = []

    lst_idx = []
    lst_lbl = None

    df = pd.read_csv(TEST_IDX_FILE)
    for path_img, rows, cols in tqdm(zip(df['path_img'], df['out_rows'], df['out_cols'])):
        abs_path_img = os.path.join(TEST_DIR, path_img)
        test_img = skio.imread(abs_path_img)
        mask = predict_mask_from_patches(model, test_img, STRIDE, (PATCH_SIZE, PATCH_SIZE), X_mean, X_std, rows, cols)
        path_fake_mask = os.path.join(TEST_DIR, path_img.replace('.png', '_msk_fake.png'))
        skio.imsave(path_fake_mask, mask)

        file_id = os.path.splitext(os.path.basename(abs_path_img))[0]
        mask_flatten = mask.flatten()
        lst_idx += ['{}_{}'.format(file_id, xx) for xx in range(mask_flatten.shape[0])]
        if lst_lbl is None:
            lst_lbl = mask_flatten
        else:
            lst_lbl = np.concatenate((lst_lbl, mask_flatten))

    csvData = pd.DataFrame(data={
        'idx': lst_idx,
        'msk': lst_lbl
    }, columns=['idx', 'msk'])
    foutCSV = '{}_predict_fake.csv'.format(TEST_IDX_FILE)
    csvData.to_csv(foutCSV, sep=',', index=None)
    foutCSV_Zip = '{}.zip'.format(foutCSV)
    with zipfile.ZipFile(foutCSV_Zip, 'w', compression=zipfile.ZIP_DEFLATED) as myzip:
        myzip.write(foutCSV, arcname=os.path.basename(foutCSV))

    return test_imgs

def create_model():
    fcn_vgg16 = FCN(input_shape=(PATCH_SIZE, PATCH_SIZE, 3), classes=3,
                    weights='imagenet', trainable_encoder=True)
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    fcn_vgg16.compile(optimizer=sgd, loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return fcn_vgg16

if __name__ == '__main__':
    # images, masks = read_train_dataset()
    model = create_model()
    X_mean = np.zeros((PATCH_SIZE, PATCH_SIZE, 3))
    X_std = np.ones((PATCH_SIZE, PATCH_SIZE, 3))
    images = read_test_dataset_and_predict(model, X_mean, X_std)
