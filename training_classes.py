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
    GlobalAveragePooling2D,
    Dropout
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
import datetime     
import pickle
import cv2 as cv
from tensorflow import keras


#function to build the ground truth labels for model training
#the input is the json file obtained from the TuSimple dataset containing lane coordinates
def build_output(json, zastavica):
    
    path_to_img = []
    output_1 = []
    output_2 = []
    output_3 = []
    output_4 = []
    izbaceni = 0
    cnt = 0
    lane_1_class = []
    lane_2_class = []
    lane_3_class = []
    lane_4_class = []
    
    for unos in json:

        unos_lanes = unos['lanes']
        y_samples = unos['h_samples']
        raw_file = unos['raw_file']
        klase_trake = unos['classes']
        
        if(zastavica == 1):
            print(len(unos_lanes))
        
        if (len(unos_lanes) == 4):

            
            temp = []

            if(len(unos_lanes[0]) > 48):
                unos_lanes[0] = unos_lanes[0][8:]
                unos_lanes[1] = unos_lanes[1][8:]
                unos_lanes[2] = unos_lanes[2][8:]
                unos_lanes[3] = unos_lanes[3][8:]
                y_samples = y_samples[8:]
                cnt += 1
            
            path_to_img.append(raw_file)
            
            #dodavanja klasa traka
            
            klasa1 = int(klase_trake.split()[0])
            klasa2 = int(klase_trake.split()[1])
            klasa3 = int(klase_trake.split()[2])
            klasa4 = int(klase_trake.split()[3])
            
            lane_1_class.append(klasa1)
            lane_2_class.append(klasa2)
            lane_3_class.append(klasa3)
            lane_4_class.append(klasa4)

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
        
    return path_to_img, output_1, output_2, output_3, output_4, izbaceni, cnt, lane_1_class, lane_2_class, lane_3_class, lane_4_class

def connect_list(liste):
    
    ab = itertools.chain.from_iterable(liste)
    lista = list(ab)
    
    return lista

#function to preprocess the images and build the image dataset
#path_to_img used from the json file in the TuSimple dataset
def build_image_dataset(path_to_img):
    
    dataset_images = []
    
    for image_path in path_to_img:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size = (256, 480))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr /= 255
        dataset_images.append(input_arr)
    
    return dataset_images

def scale_coordinates(output, mask):
    
    for i in range(len(output)):
        output[i] = output[i] * mask

def create_lane_points(output):
    
    lane_points = []
    
    for lane in output: 
        lane_points2 = []
        for i in range(0, len(lane), 2): 
            x = lane[i]
            y = lane[i+1]
            lane_points2.append([int(x),int(y)])
        lane_points.append(lane_points2)
    return lane_points



X_RATIO = 256 / 720
Y_RATIO = 480 / 1280


json_gt = [json.loads(line) for line in open('label_data_0313_withclasses.json')]
json_gt2 = [json.loads(line) for line in open('label_data_0531_withclasses.json')]
json_gt3 = [json.loads(line) for line in open('label_data_0601_withclasses.json')]

path_to_img_1, output_1_1, output_2_1, output_3_1, output_4_1, izbaceni_1, cnt_1, lane_1_1_class, lane_2_1_class, lane_3_1_class, lane_4_1_class = build_output(json_gt, 0)
path_to_img_2, output_1_2, output_2_2, output_3_2, output_4_2, izbaceni_2, cnt_2, lane_1_2_class, lane_2_2_class, lane_3_2_class, lane_4_2_class = build_output(json_gt2, 0)
path_to_img_3, output_1_3, output_2_3, output_3_3, output_4_3, izbaceni_3, cnt_3, lane_1_3_class, lane_2_3_class, lane_3_3_class, lane_4_3_class = build_output(json_gt3, 0)

path_to_img = connect_list([path_to_img_1, path_to_img_2, path_to_img_3])
output_1 = connect_list([output_1_1, output_1_2, output_1_3])
output_2 = connect_list([output_2_1, output_2_2, output_2_3])
output_3 = connect_list([output_3_1, output_3_2, output_3_3])
output_4 = connect_list([output_4_1, output_4_2, output_4_3])
lane_1_class = connect_list([lane_1_1_class, lane_1_2_class, lane_1_3_class])
lane_2_class = connect_list([lane_2_1_class, lane_2_2_class, lane_2_3_class])
lane_3_class = connect_list([lane_3_1_class, lane_3_2_class, lane_3_3_class])
lane_4_class = connect_list([lane_4_1_class, lane_4_2_class, lane_4_3_class])

output_1 = np.array(output_1)
output_2 = np.array(output_2)
output_3 = np.array(output_3)
output_4 = np.array(output_4)
lane_1_class = np.array(lane_1_class)
lane_2_class = np.array(lane_2_class)
lane_3_class = np.array(lane_3_class)
lane_4_class = np.array(lane_4_class)


train_dataset = build_image_dataset(path_to_img)
train_dataset = np.array(train_dataset)

MASK = np.array([X_RATIO, Y_RATIO])
MASK = np.repeat(MASK, 48)


scale_coordinates(output_1, MASK)
scale_coordinates(output_2, MASK)
scale_coordinates(output_3, MASK)
scale_coordinates(output_4, MASK)

lane_points_1 = create_lane_points(output_1)
lane_points_2 = create_lane_points(output_2)
lane_points_3 = create_lane_points(output_3)
lane_points_4 = create_lane_points(output_4)


print(train_dataset.shape)

#Crop each of the lanes and add them to the dataset for training
lane_dataset = []
lane_dataset_classes = []

for i in range(train_dataset.shape[0]):

    img_org = train_dataset[i].copy()

    r1 = cv2.boundingRect(np.array(lane_points_1[i][5:]))
    img_new = img_org[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if(img_new.shape[1] < 400):
        img_new = tf.image.resize(img_new, (153, 153))
        img_new = np.array(img_new)
        lane_dataset.append(img_new)
        lane_dataset_classes.append(int(lane_1_class[i])-1)

    r2 = cv2.boundingRect(np.array(lane_points_2[i][5:]))
    img_new = img_org[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    if(img_new.shape[1] < 400):
        img_new = tf.image.resize(img_new, (153, 153))
        img_new = np.array(img_new)
        lane_dataset.append(img_new)
        lane_dataset_classes.append(int(lane_2_class[i])-1)
        
    r3 = cv2.boundingRect(np.array(lane_points_3[i][5:]))    
    img_new = img_org[r3[1]:r3[1]+r3[3], r3[0]:r3[0]+r3[2]]
    if(img_new.shape[1] < 400):
        img_new = tf.image.resize(img_new, (153, 153))
        img_new = np.array(img_new)
        lane_dataset.append(img_new)
        lane_dataset_classes.append(int(lane_3_class[i])-1)
        
    r4 = cv2.boundingRect(np.array(lane_points_4[i][5:]))    
    img_new = img_org[r4[1]:r4[1]+r4[3], r4[0]:r4[0]+r4[2]]
    if(img_new.shape[1] < 400):
        img_new = tf.image.resize(img_new, (153, 153))
        img_new = np.array(img_new)
        lane_dataset.append(img_new)
        lane_dataset_classes.append(int(lane_4_class[i])-1)


lane_dataset = np.array(lane_dataset)
lane_dataset_classes = np.array(lane_dataset_classes)


print(lane_dataset.shape)
print(lane_dataset_classes.shape)


#Model arhitecture definition
input_shape = (153,153,3)

input_img = tf.keras.Input(shape=input_shape)
    
x = tf.keras.applications.resnet_v2.preprocess_input(input_img)

core = tf.keras.applications.ResNet50V2(
        include_top=False, #True
        weights="imagenet", 
        input_tensor=None, #input layer
        input_shape=input_shape, 
        pooling="max",
        )

core.trainable = False

x = core(x, training = False)

x = layers.Flatten(name='flatten')(x)
x = layers.Dense(4096, activation='relu', name='fc1')(x)
x = layers.Dense(4096, activation='relu', name='fc2')(x)
output = layers.Dense(7, activation='softmax', name='predictions')(x)


model = Model(inputs=input_img, outputs=output)

model.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


#Training the model
history = model.fit(
    x=lane_dataset,
    y=lane_dataset_classes,
    epochs=42,
    batch_size=32,
    callbacks=[tensorboard_callback],
#    validation_split=0.15,
    shuffle=True
)

#Saving the model history and weights

with open("trainHistoryDict_classes_all_42.pkl", "wb") as file:
    pickle.dump(history.history, file)

print("Spremio history dict")

model.save_weights("model_all_classes_ckpt")
load_status = model.load_weights("model_all_classes_ckpt")

# `assert_consumed` can be used as validation that all variable values have been
# restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
# methods in the Status object.
print(load_status.assert_consumed())


print('Spremio samo tezine!')







