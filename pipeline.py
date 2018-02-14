"""Full learning pipeline"""

__author__ = 'take5v'

import math
import os

import time
import zipfile
import numpy as np
import pandas as pd
import skimage.io as skio
from skimage.transform import resize
from tqdm import tqdm
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

import models as dl_models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATASET_DIR = '/mnt/82db778e-0496-450c-9b25-d1e50a90e476/data/remote-sensing-image-segmentation'
TRAIN_DIR = os.path.join(DATASET_DIR, '01_train')
TRAIN_IDX_FILE = os.path.join(TRAIN_DIR, 'idx-train.txt')
TEST_DIR = os.path.join(DATASET_DIR, '02_test_clean')
TEST_IDX_FILE = os.path.join(TEST_DIR, 'idx-test.txt')

PATCH_SIZE = 224
# PATCH_SIZE = 288
STRIDE = PATCH_SIZE * 2 // 4
FLIP = True
MIRROR = True
ROTATIONS = [90, 180, 270]

X_mean = 0
X_std = 0


def sliding_window(image, mask, stride=168, window_size=(224, 224), pad_mode='reflect'):
    """Extract patches according to a sliding window.

    Args:
        image (numpy array): The image to be processed.
        mask (numpy array): The mask that corresponds to the image
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).
        pad_mode (string): 'reflect' (default) Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.

    Returns:
        turple(list(x,y,w,h,patch)): turple of lists of patches with window_size dimensions
    """
    image_patches = []
    mask_patches = []
    # slide a window across the image
    for x in range(0, image.shape[0], stride):
        for y in range(0, image.shape[1], stride):
            new_image_patch = image[x:x + window_size[0], y:y + window_size[1]]
            new_mask_patch = mask[x:x + window_size[0], y:y + window_size[1]]

            if new_mask_patch.shape != window_size:
                # image padding for integer patches
                pad = window_size - np.asarray(new_image_patch.shape[:2])
                new_image_patch = np.lib.pad(
                    new_image_patch, ((0, pad[0]), (0, pad[1]), (0, 0)), pad_mode)
                new_mask_patch = np.lib.pad(
                    new_mask_patch, ((0, pad[0]), (0, pad[1])), pad_mode)

            image_patches.extend(augment_patch(new_image_patch))
            mask_patches.extend(augment_patch(new_mask_patch))
    return image_patches, mask_patches


def load_train():
    X_train = []
    y_train = []
    start_time = time.time()

    print('Read train images and extracting patches')
    df = pd.read_csv(TRAIN_IDX_FILE)
    for path_image, path_mask in tqdm(zip(df['path_img'], df['path_msk'])):
        image = skio.imread(os.path.join(TRAIN_DIR, path_image))
        mask = skio.imread(os.path.join(TRAIN_DIR, path_mask))

        image_patches, mask_patches = sliding_window(
            image, mask, STRIDE, (PATCH_SIZE, PATCH_SIZE))

        X_train.extend(image_patches)
        y_train.extend(mask_patches)

    print('Read train data and extracting patches time:\
          {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train

def augment_patch(image):
    transformed_images = [image]
    transformed_images.append(np.rot90(image, k=1))
    transformed_images.append(np.rot90(image, k=2))
    transformed_images.append(np.rot90(image, k=3))
    image = image[:, ::-1]
    transformed_images.append(np.rot90(image, k=1))
    transformed_images.append(np.rot90(image, k=2))
    transformed_images.append(np.rot90(image, k=3))
    return transformed_images


def read_and_normalize_train_data():
    X_train, y_train = load_train()

    print('Convert to numpy...')
    X_train = np.array(X_train, dtype='float') / 255
    y_train = np.array(y_train, dtype=np.uint8)

    print('Normalize train data...')
    global X_mean, X_std
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std

    num_classes = len(np.unique(y_train))
    y_train = np.array(y_train)
    # convert from 0, 127, 255 to 0, 1, 2 classes
    y_train[y_train == 127] = 1
    y_train[y_train == 255] = 2
    y_train = np.eye(num_classes)[y_train]

    return X_train, y_train


def run_cross_validation_create_models(nfolds=10):
    batch_size = 64
    epochs = 20

    train_data, train_target = read_and_normalize_train_data()

    yfull_train = {}
    kf = KFold(n_splits=nfolds, shuffle=True)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf.split(train_data, train_target):
        model = dl_models.create_unet(train_data.shape[1:], 3)
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        model_name = 'models/unet_{}.hdf5'.format(num_fold)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, verbose=1),
            ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.2, patience=3),
            TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=batch_size),
            # ModelCheckpoint(model_name, save_best_only=True)
        ]

        y_train = np.argmax(Y_train.reshape((-1, Y_train.shape[-1])), axis=1)

        class_weight = compute_class_weight(
            'balanced', np.unique(y_train), y_train)

        train_data_generator = ImageDataGenerator(
            # rotation_range=45,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # shear_range=0.2,
            channel_shift_range=0.5,
            zoom_range=0.2,
            fill_mode='reflect',
            # horizontal_flip=True,
            # vertical_flip=True
        )


        model.fit_generator(train_data_generator.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=len(X_train) / batch_size, epochs=epochs, callbacks=callbacks,
                            validation_data=(X_valid, Y_valid), class_weight=class_weight)

        # model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
        #           validation_data=(X_valid, Y_valid), callbacks=callbacks, class_weight=class_weight)

        predictions_valid = model.predict(
            X_valid, batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid.reshape((-1, 3)),
                         predictions_valid.reshape((-1, 3)))
        print('Score log_loss: ', score)
        sum_score += score * len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

        # save model
        model.save(model_name)

    score = sum_score / len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + \
        str(nfolds) + '_ep_' + str(epochs)
    print('Info string: {}'.format(info_string))

    return models


# def test_data_generator():
#     df = pd.read_csv(TEST_IDX_FILE)
#     for path_img, out_rows, out_cols in tqdm(zip(df['path_img'], df['out_rows'], df['out_cols'])):
#         image = skio.imread(os.path.join(TEST_DIR, path_img))
#         rows, cols = image.shape[:2]

#         X_test = extract_normalized_test_patches_from_image(image,
#             (PATCH_SIZE, PATCH_SIZE))
#         yield X_test, path_img, rows, cols, out_rows, out_cols


def extract_normalized_test_patches_from_image(image, window_size, local_window_percentage, pad_mode='reflect'):
    step = (local_window_percentage * np.asarray(window_size)).astype(int)
    patches = []
    # slide a window across the image
    for x in range(0, image.shape[0], step[0]):
        for y in range(0, image.shape[1], step[1]):
            new_patch = image[x:x + window_size[0], y:y + window_size[1]]
            if new_patch.shape[:2] != window_size:
                # image padding for integer patches
                pad = window_size - np.asarray(new_patch.shape[:2])
                new_patch = np.lib.pad(
                    new_patch, ((0, pad[0]), (0, pad[1]), (0, 0)), pad_mode)

            patches.append(new_patch)

    X_test = np.array(patches)

    print('Convert to numpy...')
    X_test = np.array(X_test, dtype='float') / 255

    print('Normalize test data...')
    global X_mean, X_std
    X_test = (X_test - X_mean) / X_std

    return X_test


def merge_several_folds_mean(fold_predictions):
    return np.mean(fold_predictions, axis=0)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * np.sqrt(2 * math.pi))


def construct_image_from_predicted_patches(patches, rows, cols, out_rows, out_cols, window_size, local_window_percentage):
    image = np.zeros((rows, cols, 3))

    step = (local_window_percentage * np.asarray(window_size)).astype(int)

    # reshape to 5D
    range_row = range(0, image.shape[0], step[0])
    range_col = range(0, image.shape[1], step[1])
    patches = patches.reshape((len(range_row), len(range_col)) + patches.shape[1:])

    # build window function
    window_function = np.zeros(patches.shape[2:4])
    x0, y0 = np.asarray(window_size) // 2
    radius = step[0] / math.sqrt(2)
    for i in range(0, window_size[0]):
        for j in range(0, window_size[1]):
            dist = math.sqrt((x0-i)**2 + (y0-j)**2)
            window_function[i, j] = 1. if dist < radius else 0.1

    for i, x in enumerate(range_row):
        for j, y in enumerate(range_col):
            image_patch_shape = image[x:x + window_size[0],
                                      y:y + window_size[1]].shape[:2]
            image[x:x + window_size[0], y:y + window_size[1]
                  ] += patches[i, j, :image_patch_shape[0], :image_patch_shape[1]] * window_function[:image_patch_shape[0], :image_patch_shape[1], np.newaxis]

    # reshape back to 4D
    patches = patches.reshape((patches.shape[0] * patches.shape[1],) + patches.shape[2:])

    # resize image
    prediction_mask_resized = resize(image, (out_rows, out_cols))

    prediction_mask_resized = from_probs_to_class(prediction_mask_resized)
    prediction_mask = from_probs_to_class(image)

    return prediction_mask, prediction_mask_resized


def from_probs_to_class(y):
    y = np.argmax(y, axis=2)
    y[y == 1] = 127
    y[y == 2] = 255
    return y


def run_cross_validation_process_test(models):
    batch_size = 32
    local_window_percentage = 0.5

    lst_idx = []
    lst_lbl = None

    df = pd.read_csv(TEST_IDX_FILE)
    #for X_test, path_img, rows, cols, out_rows, out_cols in test_data_generator():
    for path_img, out_rows, out_cols in zip(df['path_img'], df['out_rows'], df['out_cols']):
        image = skio.imread(os.path.join(TEST_DIR, path_img))
        rows, cols = image.shape[:2]

        X_test = extract_normalized_test_patches_from_image(image,
                                                            (PATCH_SIZE, PATCH_SIZE), local_window_percentage)

        test_predictions = []
        for i, model in enumerate(models):
            print('Start KFold number {} from {}'.format(i, len(models)))
            test_prediction = model.predict(
                X_test, batch_size=batch_size, verbose=2)
            test_predictions.append(test_prediction)

        Y_prediction = merge_several_folds_mean(test_predictions)

        mask, mask_resized = construct_image_from_predicted_patches(
            Y_prediction, rows, cols, out_rows, out_cols, (PATCH_SIZE, PATCH_SIZE), local_window_percentage)

        abs_path_img = os.path.join(TEST_DIR, path_img)
        path_mask = os.path.join(
            TEST_DIR, path_img.replace('.png', '_msk.png'))
        skio.imsave(path_mask, mask)
        path_resized_mask = os.path.join(
            TEST_DIR, path_img.replace('.png', '_msk_resized.png'))
        skio.imsave(path_resized_mask, mask_resized)

        file_id = os.path.splitext(os.path.basename(abs_path_img))[0]
        mask_flatten = mask_resized.flatten()
        lst_idx += ['{}_{}'.format(file_id, xx)
                    for xx in range(mask_flatten.shape[0])]
        if lst_lbl is None:
            lst_lbl = mask_flatten
        else:
            lst_lbl = np.concatenate((lst_lbl, mask_flatten))

    csv_data = pd.DataFrame(data={
        'idx': lst_idx,
        'msk': lst_lbl
    }, columns=['idx', 'msk'])
    fout_csv = '{}_predict_fake.csv'.format(TEST_IDX_FILE)
    csv_data.to_csv(fout_csv, sep=',', index=None)
    fout_csv_Zip = '{}.zip'.format(fout_csv)
    with zipfile.ZipFile(fout_csv_Zip, 'w', compression=zipfile.ZIP_DEFLATED) as myzip:
        myzip.write(fout_csv, arcname=os.path.basename(fout_csv))

def read_models():
    # read train_data to compute mean/std
    read_and_normalize_train_data()

    model_names = os.listdir('models')
    models = []

    for model_name in model_names:
        model = keras.models.load_model(os.path.join('models', model_name), custom_objects={"jaccard_coef": dl_models.jaccard_coef, "jaccard_coef_int": dl_models.jaccard_coef_int})
        models.append(model)

    return models

if __name__ == '__main__':
    models = run_cross_validation_create_models(5)
    # models = read_models()
    run_cross_validation_process_test(models)
