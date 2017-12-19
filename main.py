#%%
import os
import numpy as np

from data_utils import *
from io_utils import *
from vis_utils import *
from models import *

from keras_fcn import FCN
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

%load_ext autoreload
%autoreload 2

#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#%%
train_images, train_masks = read_train_dataset()
print(len(train_images))
# test_images = read_test_dataset()

#%%
X_train_all, y_train_all, X_train, y_train, X_val, y_val, X_mean, X_std = preprocess(train_images, train_masks)

#%%
print(len(X_train_all))

#%%
generator = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
)

#%%
# u-net
model = get_unet(X_train_all[0].shape[0], 3)
history = model.fit(X_train_all, y_train_all, batch_size=32, epochs=10)

#%%
fcn_vgg16 = FCN(input_shape=X_train[0].shape, classes=3,
                weights='None', trainable_encoder=True)

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

fcn_vgg16.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# history = fcn_vgg16.fit(X_train_all, y_train_all, batch_size=32, epochs=20)
batch_size = 64
epochs = 50
history = fcn_vgg16.fit_generator(generator.flow(X_train_all, y_train_all, batch_size=batch_size),
                                  steps_per_epoch=len(X_train_all) / batch_size, epochs=epochs)

#%%
plot_history(history)

#%%
y_pred = fcn_vgg16.predict(X_test[0][np.newaxis, ...])
y_pred_cls = np.argmax(y_pred, axis=3)
print(y_pred_cls.shape)

#%%
print(X_test[0][np.newaxis, ...].shape)

#%%
read_test_dataset_and_predict(model, X_mean, X_std)
