import os
import glob
import shutil

import tensorflow as tf
import tensorflow.keras.backend as K
import cv2 as cv


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


smooth = 0.0000001


def jacc_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))


if __name__ == '__main__':
    images = glob.glob(r'F:\cloud_net_new\test\*train*.png')
    labels = glob.glob(r'F:\cloud_net_new\test\*_class*.png')

    images.sort(key=lambda x: x.split('/')[-1])
    labels.sort(key=lambda x: x.split('/')[-1])

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    test_count = int(len(images))
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

    model = tf.keras.models.load_model(r'D:\result\weights_090-0.035023.h5',
                                       custom_objects={'jacc_coef': jacc_coef})

    print(model.evaluate(test_dataset))
    count = 0
    for image, mask in test_dataset.take(-1):
        pred_mask = model.predict(image)[0]
        tf.keras.preprocessing.image.array_to_img(image[0]).save('./output/true_image__{}.jpg'.format(count))
        tf.keras.preprocessing.image.array_to_img(mask[0]).save('./output/true_mask__{}.jpg'.format(count))
        pred_mask = pred_mask * 255
        pred_mask[pred_mask < 127] = 0
        pred_mask[pred_mask > 0] = 255

        cv.imwrite('./output/prediction_mask__{}.jpg'.format(count), pred_mask)
        count += 1

        if count == 200:
            break
