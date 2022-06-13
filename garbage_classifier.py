import tensorflow as tf
import tensorflow_hub as hub
# import numpy as np
# import os, path
import splitfolders
# import PIL

from tensorflow import keras

DATA_DIR = 'C:\\Users\\zenix\\Documents\\Bangkit\\cloud\\garbage_classification'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
# OUT_SHAPE = 1280
AUTO_TUNE = tf.data.AUTOTUNE

splitfolders.ratio(
    './garbage_classification',
    output="dataset",
    seed=1337,
    ratio=(0.8, 0.1, 0.1))

TRAIN_DIR = './dataset/train'
VAL_DIR = './dataset/val'
TEST_DIR = './dataset/test'

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=None)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=None)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=None)

class_names = train_ds.class_names
num_classes = len(class_names)


def preprocessing(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return image, label


train_img = train_ds.shuffle(1000).batch(BATCH_SIZE).map(preprocessing).prefetch(buffer_size=AUTO_TUNE)
val_img = val_ds.shuffle(1000).batch(BATCH_SIZE).map(preprocessing).prefetch(buffer_size=AUTO_TUNE)
test_img = test_ds.batch(BATCH_SIZE).map(preprocessing)

load_mobilenet = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

mobilenet_layer = hub.KerasLayer(load_mobilenet,
                               input_shape=IMAGE_SIZE + (3,),
                               output_shape=[1280],
                               trainable=False)

model = tf.keras.Sequential([
    mobilenet_layer,
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.build(input_shape=(None, 224, 224, 3))

model.summary()

optimizer_param = keras.optimizers.Adam(learning_rate=0.001)
loss_param = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics_param = ['accuracy']

model.compile(optimizer=optimizer_param,
              loss=loss_param,
              metrics=metrics_param)

EPOCHS = 15

history = model.fit(train_img,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=val_img)

model.save("classifier.h5")

print("Evaluating Model:")
model.evaluate(test_img,
               verbose=1)

eval_model = keras.models.load_model("classifier.h5")
eval_model.evaluate(test_img,
                    verbose=1)
