
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from glob import glob

import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

import configuration as config
from network import contrast_net
from utils import random_jpeg_augmentation
from keras import backend as K

import matplotlib.pyplot as plt

import keras

# GPU management
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
K.set_session(sess)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"


def data_generators():

    color_space = 'rgb' if config.PATCH_CHANNELS == 3 else 'grayscale'

    # Augment by randomly JPEG-compressing each image in batch (data is also scaled by 1/255)
    if config.JPEG_AUGMENTATION:
        gen_training = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=random_jpeg_augmentation)
        gen_validation = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=random_jpeg_augmentation)
        gen_test = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=random_jpeg_augmentation)

    # No augmentation, simply scale data
    else:
        gen_training = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
        gen_validation = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
        gen_test = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

    train_g = gen_training.flow_from_directory(directory=config.TRAIN_FOLDER,
                                               target_size=(config.PATCH_SIZE, config.PATCH_SIZE),
                                               batch_size=config.TRAIN_BATCH, color_mode=color_space,
                                               classes=[config.CLASS_0_TAG, config.CLASS_1_TAG],
                                               class_mode='categorical')

    val_g = gen_validation.flow_from_directory(directory=config.VALIDATION_FOLDER,
                                               target_size=(config.PATCH_SIZE, config.PATCH_SIZE),
                                               batch_size=config.VALIDATION_BATCH,
                                               color_mode=color_space, class_mode='categorical',
                                               classes=[config.CLASS_0_TAG, config.CLASS_1_TAG])

    test_g = gen_test.flow_from_directory(directory=config.TEST_FOLDER,
                                          target_size=(config.PATCH_SIZE, config.PATCH_SIZE),
                                          batch_size=config.TEST_BATCH, color_mode=color_space,
                                          classes=[config.CLASS_0_TAG, config.CLASS_1_TAG],
                                          class_mode='categorical')

    print(train_g.class_indices)
    print(val_g.class_indices)
    print(test_g.class_indices)

    return train_g, val_g, test_g


def count_patches():

    """ Get the number of training, validation and test examples. Assuming same number of examples per class
    """

    n_val = len(glob(os.path.join(config.VALIDATION_FOLDER, config.CLASS_0_TAG, '*.*')))
    n_test = len(glob(os.path.join(config.TEST_FOLDER, config.CLASS_0_TAG, '*.*')))
    n_train = len(glob(os.path.join(config.TRAIN_FOLDER, config.CLASS_0_TAG, '*.*')))

    return 2*n_train, 2*n_val, 2*n_test


if __name__ == '__main__':

    # MODEL SETUP / MODEL RESUMING

    if config.RESUME_TRAINING:
        model = keras.models.load_model(config.RESUME_MODEL)
        print('Resuming training from model {}'.format(config.RESUME_MODEL))
    else:
        model = contrast_net(in_shape=(config.PATCH_SIZE, config.PATCH_SIZE, config.PATCH_CHANNELS), nf_base=32)

        #model.compile(loss=keras.losses.categorical_crossentropy,
         #             optimizer=keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.9999, epsilon=1e-08),
          #            metrics=['accuracy'])

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.9999, epsilon=1e-08),
                      metrics=['accuracy'])

    # Display a summary of configuration and network
    print(model.summary())
    plot_model(model, to_file='resizenet.png', show_shapes=True)

    train_generator, val_generator, test_generator = data_generators()

    n_train_imgs, n_val_imgs, n_test_imgs = count_patches()

    if not os.path.exists(os.path.join(config.MODEL_FOLDER, 'checkpoints')):
        os.makedirs(os.path.join(config.MODEL_FOLDER, 'checkpoints'))

    checkpoint_saver = ModelCheckpoint(filepath=os.path.join(config.MODEL_FOLDER, 'checkpoints',
                                                             'ckpt.epoch{epoch:02d}-loss{val_loss:.2f}.h5'),
                                       monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=False)

    # MODEL TRAINING

    begin_time = time.time()
    data = model.fit_generator(generator=train_generator, steps_per_epoch=int(n_train_imgs / config.TRAIN_BATCH),
                        epochs=config.NUM_EPOCHS,  validation_data=val_generator,
                        validation_steps=int(n_val_imgs / config.VALIDATION_BATCH), callbacks=[checkpoint_saver])

    #plt.plot(data.history['val_acc'], 'r', data.history['acc'], 'b')
    #plt.xlabel('epoch')
    #plt.ylabel('accuracy')
    #plt.show()


    # plt.savefig(params.results_dir + '/' + '_accuracy.jpg')

    # Save model
    if config.JPEG_AUGMENTATION:
        model_file = 'model_ICIPNet_median_RandomizationProject_4epoch.h5'
    else:
        model_file = 'model_ICIPNet_Resize_RandomizationProject_4epoch.h5'
    model.save(os.path.join(config.MODEL_FOLDER, model_file))

    print('Training ended. Elapsed time: {} seconds'.format(time.time()-begin_time))

    # MODEL TESTING

    score = model.evaluate_generator(test_generator, steps=int(n_test_imgs / config.TEST_BATCH))

    print('Test accuracy {:g}. Loss {:g}'.format(score[1], score[0]))

    print(data.history.keys())
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(data.history['acc'])
    plt.plot(data.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()
