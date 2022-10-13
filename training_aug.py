import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input,
    Add,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    GlobalMaxPooling2D,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
import datetime     
import pickle
#from tensorflow.keras.optimizers import Adam
#from keras.optimizers import adam
from tensorflow import keras

def build_output(json, zastavica):

    path_to_img = []
    output_1 = []
    output_2 = []
    output_3 = []
    output_4 = []
    izbaceni = 0
    cnt = 0

    for unos in json:

        unos_lanes = unos["lanes"]
        y_samples = unos["h_samples"]
        raw_file = unos["raw_file"]


        if len(unos_lanes) == 4:

            temp = []

            if len(unos_lanes[0]) > 48:
                unos_lanes[0] = unos_lanes[0][8:]
                unos_lanes[1] = unos_lanes[1][8:]
                unos_lanes[2] = unos_lanes[2][8:]
                unos_lanes[3] = unos_lanes[3][8:]
                y_samples = y_samples[8:]
                cnt += 1

            path_to_img.append(raw_file)

            for x_1, y_1 in zip(unos_lanes[0], y_samples):
                temp.append(x_1)
                temp.append(y_1)
            output_1.append(temp)

            temp = []
            for x_2, y_2 in zip(unos_lanes[1], y_samples):
                temp.append(x_2)
                temp.append(y_2)
            output_2.append(temp)

            temp = []
            for x_3, y_3 in zip(unos_lanes[2], y_samples):
                temp.append(x_3)
                temp.append(y_3)
            output_3.append(temp)

            temp = []
            for x_4, y_4 in zip(unos_lanes[3], y_samples):
                temp.append(x_4)
                temp.append(y_4)
            output_4.append(temp)
        else:
            izbaceni += 1

    return path_to_img, output_1, output_2, output_3, output_4, izbaceni, cnt


def connect_list(liste):

    ab = itertools.chain.from_iterable(liste)
    lista = list(ab)

    return lista


def build_image_dataset(path_to_img):

    dataset_images = []

    for image_path in path_to_img:
        image = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(256, 480)
        )
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr /= 255
        dataset_images.append(input_arr)

    return dataset_images


def scale_coordinates(output, mask):

    for i in range(len(output)):
        output[i] = output[i] * mask


def custom_loss(y_true, y_pred):
    return tf.reduce_sum(tf.abs(tf.subtract(y_true, y_pred)))

def create_lane_points(output):
    
    lane_points = []
    
    for lane in output: 
        lane_points2 = []
        for i in range(0, len(lane), 2): 
            x = lane[i]
            y = lane[i+1]
            lane_points2.append([x, y]) 
        lane_points.append(lane_points2)
    return lane_points

def rotate_image(image_list, angle): 
    rotated_images = []
    
    rows = image_list.shape[1]
    cols = image_list.shape[2]
    # cols-1 and rows-1 are the coordinate limits.
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angle,1)
    
    for img in image_list:
        dst = cv2.warpAffine(img,M,(cols,rows))
        rotated_images.append(dst)
        
    return rotated_images

def rotate_points(points, angle):
    new_points = []
    M = cv2.getRotationMatrix2D(((256-1)/2.0,(480-1)/2.0),angle,1) 
    for img in points:
        ones = np.ones(shape=(len(img), 1))
        points_ones = np.hstack([img, ones])
        transformed_points = M.dot(points_ones.T).T
        new_points.append(transformed_points)
    return new_points


input_shape = (256, 480, 3)

input_img = tf.keras.Input(shape=input_shape)

x = tf.keras.applications.resnet_v2.preprocess_input(input_img)

core = tf.keras.applications.ResNet50V2(
        include_top=False, 
        weights="imagenet",
        input_tensor=None, 
        input_shape=input_shape, 
        pooling="max",
        )

core.trainable = False

x = core(x, training = False)

branch_1 = Dense(120, kernel_initializer = glorot_uniform(seed=0), activation = 'relu')(x)
branch_2 = Dense(120, kernel_initializer = glorot_uniform(seed=0), activation = 'relu')(x)
branch_3 = Dense(120, kernel_initializer = glorot_uniform(seed=0), activation = 'relu')(x)
branch_4 = Dense(120, kernel_initializer = glorot_uniform(seed=0), activation = 'relu')(x)

branch_1 = Dense(96, activation = 'linear')(branch_1)
branch_2 = Dense(96, activation = 'linear')(branch_2)
branch_3 = Dense(96, activation = 'linear')(branch_3)
branch_4 = Dense(96, activation = 'linear')(branch_4)

model = Model(inputs=input_img, outputs=[branch_1, branch_2, branch_3, branch_4])


def model_aug(input_shape):

    input_img = tf.keras.Input(shape=input_shape)

    x = BatchNormalization()(input_img)

    x = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    x = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    x = MaxPooling2D()(x)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    x = MaxPooling2D()(x)

    x = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    x = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    x = MaxPooling2D()(x)

    x = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    x = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    x = MaxPooling2D()(x)

    x = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    x = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    x = Flatten()(x)

    branch_1 = Dense(120, kernel_initializer=glorot_uniform(seed=0), activation="relu")(
        x
    )
    branch_2 = Dense(120, kernel_initializer=glorot_uniform(seed=0), activation="relu")(
        x
    )
    branch_3 = Dense(120, kernel_initializer=glorot_uniform(seed=0), activation="relu")(
        x
    )
    branch_4 = Dense(120, kernel_initializer=glorot_uniform(seed=0), activation="relu")(
        x
    )

    branch_1 = Dense(96, activation="linear")(branch_1)
    branch_2 = Dense(96, activation="linear")(branch_2)
    branch_3 = Dense(96, activation="linear")(branch_3)
    branch_4 = Dense(96, activation="linear")(branch_4)

    model = Model(inputs=input_img, outputs=[branch_1, branch_2, branch_3, branch_4])

    return model

json_gt = [json.loads(line) for line in open('label_data_0313.json')]
json_gt2 = [json.loads(line) for line in open('label_data_0531.json')]
json_gt3 = [json.loads(line) for line in open('label_data_0601.json')]

(
    path_to_img_1,
    output_1_1,
    output_2_1,
    output_3_1,
    output_4_1,
    izbaceni_1,
    cnt_1,
) = build_output(json_gt, 0)
(
    path_to_img_2,
    output_1_2,
    output_2_2,
    output_3_2,
    output_4_2,
    izbaceni_2,
    cnt_2,
) = build_output(json_gt2, 0)

(
    path_to_img_3,
    output_1_3,
    output_2_3,
    output_3_3,
    output_4_3,
    izbaceni_3,
    cnt_3,
) = build_output(json_gt3, 0)

path_to_img = connect_list([path_to_img_1, path_to_img_2, path_to_img_3])
output_1 = connect_list([output_1_1, output_1_2, output_1_3])
output_2 = connect_list([output_2_1, output_2_2, output_2_3])
output_3 = connect_list([output_3_1, output_3_2, output_3_3])
output_4 = connect_list([output_4_1, output_4_2, output_4_3])

output_1 = np.array(output_1)
output_2 = np.array(output_2)
output_3 = np.array(output_3)
output_4 = np.array(output_4)

X_RATIO = 256 / 720
Y_RATIO = 480 / 1280
MASK = np.array([X_RATIO, Y_RATIO])
MASK = np.repeat(MASK, 48)

scale_coordinates(output_1, MASK)
scale_coordinates(output_2, MASK)
scale_coordinates(output_3, MASK)
scale_coordinates(output_4, MASK)



train_dataset_lista = build_image_dataset(path_to_img)
train_dataset_lista.extend(rotate_image(np.array(train_dataset_lista[1600:]), 15))

train_dataset = np.array(train_dataset_lista)




lane_points_1 = create_lane_points(output_1[1600:])
lane_points_2 = create_lane_points(output_2[1600:])
lane_points_3 = create_lane_points(output_3[1600:])
lane_points_4 = create_lane_points(output_4[1600:])


lane_points_1_a = np.array(rotate_points(lane_points_1, 15))
lane_points_2_a = np.array(rotate_points(lane_points_2, 15))
lane_points_3_a = np.array(rotate_points(lane_points_3, 15))
lane_points_4_a = np.array(rotate_points(lane_points_4, 15))


lane_points_1_a = np.reshape(lane_points_1_a, (1382,96))
lane_points_2_a = np.reshape(lane_points_1_a, (1382,96))
lane_points_3_a = np.reshape(lane_points_1_a, (1382,96))
lane_points_4_a = np.reshape(lane_points_1_a, (1382,96))



output_1 = np.concatenate((output_1, lane_points_1_a))
output_2 = np.concatenate((output_2, lane_points_2_a))
output_3 = np.concatenate((output_3, lane_points_3_a))
output_4 = np.concatenate((output_4, lane_points_4_a))

print(train_dataset.shape)
print(output_1.shape)
print(output_2.shape)
print(output_3.shape)
print(output_4.shape)

del train_dataset_lista
del lane_points_1
del lane_points_2
del lane_points_3
del lane_points_4
del lane_points_1_a
del lane_points_2_a
del lane_points_3_a
del lane_points_4_a


print("Stvaram model")

model.compile(optimizer=keras.optimizers.Adam(), loss=custom_loss, metrics=["accuracy"])

output_1 = tf.cast(output_1, tf.float32)
output_2 = tf.cast(output_2, tf.float32)
output_3 = tf.cast(output_3, tf.float32)
output_4 = tf.cast(output_4, tf.float32)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


print("Trening!")

history = model.fit(
    x=train_dataset,
    y=[output_1, output_2, output_3, output_4],
    epochs=130,
    batch_size=1,
    callbacks=[tensorboard_callback],
    validation_split=0.1,
    shuffle=True
)



print("Gotov trening!")

#core.trainable = True

#for layer in core.layers[:55]:
#    layer.trainable = False

#model.compile(optimizer=keras.optimizers.Adam(1e-5), loss=custom_loss, metrics=["accuracy"])

#history_fine = model.fit(
#        x = train_dataset,
#        y = [output_1, output_2, output_3, output_4],
#        epochs = 140,
#        initial_epoch = history.epoch[-1],
#        callbacks=[tensorboard_callback],
#        validation_split=0.15,
#        shuffle=True)


del train_dataset

with open("trainHistoryDict130_augmentation_pretrain_model.pkl", "wb") as file:
    pickle.dump(history.history, file)

print("Spremio history")



model.save_weights("pre_model170_aug_ckpt")
load_status = model.load_weights("pre_model170_aug_ckpt")

# `assert_consumed` can be used as validation that all variable values have been
# restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
# methods in the Status object.
print(load_status.assert_consumed())


print('Spremio samo tezine!')
