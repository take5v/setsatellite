"""This module does blah blah."""

import numpy as np
import skimage.transform
from skimage.transform import resize


def preprocess(train_img, train_mask):
    X_train = np.array(train_img, dtype='float') / 255
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std

    num_classes = len(np.unique(train_mask))
    y_train = np.array(train_mask)
    # convert from 0, 127, 255 to 0, 1, 2 classes
    y_train[y_train == 127] = 1
    y_train[y_train == 255] = 2
    y_train = np.eye(num_classes)[y_train]

    train_elements = round(len(X_train) * 0.8)
    X_train_all = X_train
    y_train_all = y_train
    X_val = X_train_all[train_elements:]
    y_val = y_train_all[train_elements:]
    X_train = X_train_all[:train_elements]
    y_train = y_train_all[:train_elements]

    return X_train_all, y_train_all, X_train, y_train, X_val, y_val, X_mean, X_std


def preprocess_test(test_img, X_mean, X_std):
    X_train = np.array(test_img, dtype='float') / 255
    X_train = (X_train - X_mean) / X_std

    return X_train


def sliding_window_with_coords(image, stride=10, window_size=(20,20)):
    """Extract patches according to a sliding window.

    Args:
        image (numpy array): The image to be processed.
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).

    Returns:
        list(x,y,w,h,patch): list of patches and coordinates with window_size dimensions
    """
    patches = []
    # slide a window across the image
    for x in range(0, image.shape[0], stride):
        for y in range(0, image.shape[1], stride):
            new_patch = image[x:x + window_size[0], y:y + window_size[1]]
            if new_patch.shape[:2] == window_size:
                patches.append((x, y, window_size[0], window_size[1], new_patch))
    return patches


def predict_mask_from_patches(model, image, stride, patch_size, X_mean, X_std, rows, cols):
    patches = sliding_window_with_coords(image, stride, patch_size)

    votes = np.zeros(image.shape[:2] + (3,))
    patches_count = np.ones(image.shape[:2] + (1,))

    for x,y,w,h,patch in patches:
        preprocessed_patch = preprocess_test(patch[np.newaxis, ...], X_mean, X_std)
        y_pred = model.predict(preprocessed_patch)
        votes[x:x + w, y:y + h] = y_pred
        patches_count[x:x + w, y:y + h] += 1

    prediction_mask = votes / patches_count

    print('prediction_mask shape = ', prediction_mask.shape)
    # resize image
    prediction_mask_resized = resize(prediction_mask, (rows, cols))

    prediction_mask_resized = np.argmax(prediction_mask_resized, axis=2)
    prediction_mask_resized[prediction_mask_resized == 1] = 127
    prediction_mask_resized[prediction_mask_resized == 2] = 255

    return prediction_mask_resized


def sliding_window(image, stride=10, window_size=(20,20)):
    """Extract patches according to a sliding window.

    Args:
        image (numpy array): The image to be processed.
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).

    Returns:
        list: list of patches with window_size dimensions
    """
    patches = []
    # slide a window across the image
    for x in range(0, image.shape[0], stride):
        for y in range(0, image.shape[1], stride):
            new_patch = image[x:x + window_size[0], y:y + window_size[1]]
            if new_patch.shape[:2] == window_size:
                patches.append(new_patch)
    return patches

def transform(patch, flip=False, mirror=False, rotations=[]):
    """Perform data augmentation on a patch.

    Args:
        patch (numpy array): The patch to be processed.
        flip (bool, optional): Up/down symetry.
        mirror (bool, optional): left/right symetry.
        rotations (int list, optional) : rotations to perform (angles in deg).

    Returns:
        array list: list of augmented patches
    """
    transformed_patches = [patch]
    for angle in rotations:
        transformed_patches.append(skimage.img_as_ubyte(skimage.transform.rotate(patch, angle)))
    if flip:
        transformed_patches.append(np.flipud(patch))
    if mirror:
        transformed_patches.append(np.fliplr(patch))
    return transformed_patches


def augmented_sliding_window(patches, flip=False, mirror=False, rotations=[]):
    transformed_patches = []

    for patch in patches:
        transformed_patches.extend(transform(patch, flip, mirror, rotations))

    return transformed_patches

def extract_augmented_patches_from_image(image, stride, window_size, flip, mirror, rotations):
    patches = sliding_window(image, stride=stride, window_size=window_size)
    augmented_patches = augmented_sliding_window(patches, flip, mirror, rotations)

    return augmented_patches

