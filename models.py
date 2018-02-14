"""Here it will be different models for semantic segmentation"""

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Concatenate, concatenate, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

smooth = 1e-12


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def create_unet_2(input_shape, number_of_classes):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(number_of_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model


def create_unet(input_shape, number_of_classes):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(number_of_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    # optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model

def create_unet_3(input_shape, number_of_classes):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(number_of_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    # optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model



def dense_block(layers_count, filters, previous_layer, model_layers, level):
    model_layers[level] = {}
    for i in range(layers_count):
        model_layers[level]['b_norm' + str(i + 1)] = BatchNormalization(mode=0, axis=3,
                                                                        gamma_regularizer=l2(
                                                                            0.0001),
                                                                        beta_regularizer=l2(0.0001))(previous_layer)
        model_layers[level]['act' + str(i + 1)] = Activation(
            'relu')(model_layers[level]['b_norm' + str(i + 1)])
        model_layers[level]['conv' + str(i + 1)] = Conv2D(filters, kernel_size=(3, 3), padding='same',
                                                          kernel_initializer="he_uniform",
                                                          data_format='channels_last')(model_layers[level]['act' + str(i + 1)])
        model_layers[level]['drop_out' +
                            str(i + 1)] = Dropout(0.2)(model_layers[level]['conv' + str(i + 1)])
        previous_layer = model_layers[level]['drop_out' + str(i + 1)]
    # print(model_layers)
    # return last layer of this level
    return model_layers[level]['drop_out' + str(layers_count)]


def transition_down(filters, previous_layer, model_layers, level):
    model_layers[level] = {}
    model_layers[level]['b_norm'] = BatchNormalization(mode=0, axis=3,
                                                       gamma_regularizer=l2(
                                                           0.0001),
                                                       beta_regularizer=l2(0.0001))(previous_layer)
    model_layers[level]['act'] = Activation(
        'relu')(model_layers[level]['b_norm'])
    model_layers[level]['conv'] = Conv2D(filters, kernel_size=(1, 1), padding='same',
                                         kernel_initializer="he_uniform")(model_layers[level]['act'])
    model_layers[level]['drop_out'] = Dropout(0.2)(model_layers[level]['conv'])
    model_layers[level]['max_pool'] = MaxPooling2D(pool_size=(2, 2),
                                                   strides=(2, 2),
                                                   data_format='channels_last')(model_layers[level]['drop_out'])
    return model_layers[level]['max_pool']


def transition_up(filters, input_shape, output_shape, previous_layer, model_layers, level):
    model_layers[level] = {}
    model_layers[level]['conv'] = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2),
                                                  padding='same',
                                                  output_shape=output_shape,
                                                  input_shape=input_shape,
                                                  kernel_initializer="he_uniform",
                                                  data_format='channels_last')(previous_layer)

    return model_layers[level]['conv']


def get_tiramisu(input_size, number_of_classes, number_of_channels=3):
    inputs = Input((input_size, input_size, number_of_channels))

    first_conv = Conv2D(48, kernel_size=(3, 3), padding='same',
                        kernel_initializer="he_uniform",
                        kernel_regularizer=l2(0.0001),
                        data_format='channels_last')(inputs)

    enc_model_layers = {}


    layer_1_down = dense_block(5, 108, first_conv, enc_model_layers, 'layer_1_down')  # 5*12 = 60 + 48 = 108
    layer_1a_down = transition_down(108, layer_1_down, enc_model_layers,  'layer_1a_down')

    layer_2_down = dense_block(5, 168, layer_1a_down, enc_model_layers, 'layer_2_down')  # 5*12 = 60 + 108 = 168
    layer_2a_down = transition_down(168, layer_2_down, enc_model_layers,  'layer_2a_down')

    layer_3_down = dense_block(5, 228, layer_2a_down, enc_model_layers, 'layer_3_down')  # 5*12 = 60 + 168 = 228
    layer_3a_down = transition_down(228, layer_3_down, enc_model_layers,  'layer_3a_down')

    layer_4_down = dense_block(5, 288, layer_3a_down, enc_model_layers, 'layer_4_down')  # 5*12 = 60 + 228 = 288
    layer_4a_down = transition_down(288, layer_4_down, enc_model_layers,  'layer_4a_down')

    layer_5_down = dense_block(5, 348, layer_4a_down, enc_model_layers, 'layer_5_down')  # 5*12 = 60 + 288 = 348
    layer_5a_down = transition_down(348, layer_5_down, enc_model_layers,  'layer_5a_down')

    layer_bottleneck = dense_block(15, 408, layer_5a_down, enc_model_layers,  'layer_bottleneck')  # m = 348 + 5*12 = 408

    # m = 348 + 5x12 + 5x12 = 468.
    layer_1_up = transition_up(468, (468, 7, 7), (None, 468, 14, 14), layer_bottleneck, enc_model_layers, 'layer_1_up')
    skip_up_down_1 = concatenate([layer_1_up, enc_model_layers['layer_5_down']['conv' + str(5)]], axis=-1)
    layer_1a_up = dense_block(5, 468, skip_up_down_1, enc_model_layers, 'layer_1a_up')

    layer_2_up = transition_up(408, (408, 14, 14), (None, 408, 28, 28),
                                   layer_1a_up, enc_model_layers, 'layer_2_up')  # m = 288 + 5x12 + 5x12 = 408
    skip_up_down_2 = concatenate([layer_2_up, enc_model_layers['layer_4_down']['conv' + str(5)]], axis=-1)
    layer_2a_up = dense_block(5, 408, skip_up_down_2, enc_model_layers, 'layer_2a_up')

    layer_3_up = transition_up(348, (348, 28, 28), (None, 348, 56, 56),
                                   layer_2a_up, enc_model_layers, 'layer_3_up')  # m = 228 + 5x12 + 5x12 = 348
    skip_up_down_3 = concatenate([layer_3_up, enc_model_layers['layer_3_down']['conv' + str(5)]], axis=-1)
    layer_3a_up = dense_block(5, 348, skip_up_down_3, enc_model_layers, 'layer_3a_up')

    layer_4_up = transition_up(288, (288, 56, 56), (None, 288, 112, 112),
                                   layer_3a_up, enc_model_layers, 'layer_4_up')  # m = 168 + 5x12 + 5x12 = 288
    skip_up_down_4 = concatenate(
        [layer_4_up, enc_model_layers['layer_2_down']['conv' + str(5)]], axis=-1)
    layer_4a_up = dense_block(
        5, 288, skip_up_down_4, enc_model_layers, 'layer_4a_up')

    layer_5_up = transition_up(228, (228, 112, 112), (None, 228, 224, 224),
                                   layer_4a_up, enc_model_layers, 'layer_5_up')  # m = 108 + 5x12 + 5x12 = 228
    skip_up_down_5 = concatenate(
        [layer_5_up, enc_model_layers['layer_1_down']['conv' + str(5)]], axis=-1)
    layer_5a_up = dense_block(
        5, 228, skip_up_down_5, enc_model_layers, 'concatenate')

    # last
    last_conv = Conv2D(number_of_classes, activation='linear',
                       kernel_size=(1,1),
                       padding='same',
                       kernel_regularizer = l2(0.0001),
                       data_format='channels_last')(layer_5a_up)

    reshape = Reshape((number_of_classes, input_size * input_size))(last_conv)
    perm = Permute((2, 1))(reshape)
    act = Activation('softmax')(perm)

    model = Model(inputs=[inputs], outputs=[act])

    return model
