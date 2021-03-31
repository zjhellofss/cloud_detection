import os
import glob
import shutil

import pix2pix
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping


def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    return img


def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img


max_possible_input_value = 255


def normalize(input_image, input_mask):
    input_image /= 255
    input_mask /= 255
    return input_image, input_mask


def load_image(input_image_path, input_mask_path):
    input_image = read_jpg(input_image_path)
    input_mask = read_png(input_mask_path)
    input_image = tf.image.resize(input_image, (384, 384))
    input_mask = tf.image.resize(input_mask, (384, 384))
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def load_image_test(input_image_path, input_mask_path):
    input_image = read_jpg(input_image_path)
    input_mask = read_png(input_mask_path)

    input_image = tf.image.resize(input_image, (384, 384))
    input_mask = tf.image.resize(input_mask, (384, 384))

    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


if __name__ == '__main__':


    image_paths = './dataset/JPEGImages/'
    label_paths = './dataset/Annotations'


    images = glob.glob(image_paths + '*.png')
    labels = glob.glob(label_paths + '*.png')
    images.sort(key=lambda x: x.split('/')[-1])
    labels.sort(key=lambda x: x.split('/')[-1])

    np.random.seed(2022)
    index = np.random.permutation(len(images))

    images = np.array(images)[index]
    labels = np.array(labels)[index]

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    test_count = int(len(images) * 0.2)
    train_count = len(images) - test_count

    dataset_train = dataset.skip(test_count)
    dataset_test = dataset.take(test_count)

    BATCH_SIZE = 16
    BUFFER_SIZE = 331
    STEPS_PER_EPOCH = train_count // BATCH_SIZE
    VALIDATION_STEPS = test_count // BATCH_SIZE
    ############################################
    train = dataset_train.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset_test.map(load_image_test)

    train_dataset = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(batch_size=BATCH_SIZE)
    OUTPUT_CHANNELS = 1

    from unet import _unet
    from resnet import get_resnet50_encoder

    model = _unet(1, get_resnet50_encoder)
    starting_learning_rate = 1e-4
    # model.compile(optimizer=Adam(lr=starting_learning_rate), loss=jacc_coef, metrics=[jacc_coef, 'accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=starting_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                           ])
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=2)
    EPOCHS = 100
    weight_path = r'./models'
    weights_path = os.path.join(weight_path, 'weights_{epoch:03d}-{val_loss:.6f}.h5')
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    starting_learning_rate = 1e-4
    end_learning_rate = 1e-8
    patience = 15
    decay_factor = 0.7
    lr_reducer = ReduceLROnPlateau(factor=decay_factor, cooldown=0, patience=patience, min_lr=end_learning_rate,
                                   verbose=1)

    csv_logger = CSVLogger('Cloud' + '_log_1.log')
    # display([sample_image, sample_mask])
    from utils import ADAMLearningRateTracker

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_dataset,
                              callbacks=[model_checkpoint, csv_logger, lr_reducer,
                                         ADAMLearningRateTracker(end_learning_rate), early_stopping]
                              )
