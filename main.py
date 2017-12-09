#%%
import os
import numpy as np

from data_utils import *
from io_utils import *
from vis_utils import *

from keras_fcn import FCN
from keras import optimizers

%load_ext autoreload
%autoreload 2

#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#%%
train_images, train_masks = read_train_dataset()
# test_images = read_test_dataset()

#%%
X_train_all, y_train_all, X_train, y_train, X_val, y_val, X_mean, X_std = preprocess(train_images, train_masks)

#%%
fcn_vgg16 = FCN(input_shape=X_train[0].shape, classes=3,
                weights='None', trainable_encoder=True)

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

fcn_vgg16.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history = fcn_vgg16.fit(X_train_all, y_train_all, batch_size=32, epochs=10)

#%%
plot_history(history)

#%%
y_pred = fcn_vgg16.predict(X_test[0][np.newaxis, ...])
y_pred_cls = np.argmax(y_pred, axis=3)
print(y_pred_cls.shape)

#%%
print(X_test[0][np.newaxis, ...].shape)

#%%
read_test_dataset_and_predict(fcn_vgg16, X_mean, X_std)
