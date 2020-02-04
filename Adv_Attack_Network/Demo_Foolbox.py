import foolbox
from foolbox.models import KerasModel
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras.datasets import mnist
#from scipy.misc import imread
from imageio import imread
from glob import glob
#from utils import force_linear_activation
import cv2
from PIL import Image
import math
import tensorflow as tf
import os
from keras.models import load_model,Model
from glob import glob
from keras.preprocessing.image import img_to_array, load_img



def show_figures(I, Z, true_score,adv_score):
    plt.figure()

    true_class = np.argmax(true_score)
    adv_class = np.argmax(adv_score)

    plt.subplot(1, 3, 1)
    plt.title('Original (class {}, score {:2.2f})'.format(true_class, true_score[true_class]))
    plt.imshow(I, cmap=plt.get_cmap('gray'))  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial (class {}, score {:2.2f})'.format(adv_class, adv_score[adv_class]))
    plt.imshow(Z, cmap=plt.get_cmap('gray'))  # ::-1 to convert BGR to RGB
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = np.double(Z) - np.double(I)  # Z - I
    plt.imshow(difference,cmap=plt.get_cmap('Blues'))# / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')

    plt.show()
    return

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':

    def main():

        # Load Keras model
        model = load_model(r'.................................................h5')


        Ptype = 'probabilities' # (default) with the softmax
        # # Switch softmax with linear activations -- to avoid the softmax
        #model = force_linear_activation(model=model, savemodel=None)
        #Ptype = 'logits'

        compressJPEG  = 0 #'true'
        jpeg = 0
        jpeg_quality = 85

        # size (no color images)
        img_rows, img_cols, img_chans = 64,64, 1
        num_classes = 2

        #---------------------------------------------------------
        #  Load test data, define labels, test the model
        #-----------------------------------------------------------

      
        images = glob(r'F:..................................\*.png')

        label = 1 # label = 0 for Manipulated, 1 for Original ------ for StammNets, it is the reverse ! (0 for Original)

        #number of imagess for testing the model
        #numImg = len(images) # <= len(images)

        numImg=100

        #np.random.seed(1234)
        #index = np.random.randint(len(images), size=numImg)
        index = np.arange(numImg)

        x_test = np.zeros((numImg, img_rows, img_cols))
        for i in np.arange(numImg):
            img = imread(images[index[i]])  # Flatten=True means convert to gray on the fly
            if compressJPEG:
                img1 = Image.fromarray(img)
                img1.save('temp.jpeg', "JPEG", quality=jpeg_quality)
                img = Image.open('temp.jpeg')
            x_test[i] = img

        # Labels
        y_test_c = np.tile(label, numImg)

        # Convert labels to one-hot with Keras
        y_test = keras.utils.to_categorical(y_test_c, num_classes)

        # Reshape test data, divide by 255 because net was trained this way
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chans)

        x_test = x_test.astype('float32')
        x_test /= 255

        # Test legitimate examples
        score = model.evaluate(x_test, y_test, verbose=0)
        predicted_legitimate_labels = np.argmax(model.predict(x_test), axis=1)
        print('Accuracy on legitimate images (all): {:3.4f}'.format(score[1]))

        # ----------------------------------------------------------------------------------------------------------------------
        # Attack the [correctly classified] images in the test set
        # ----------------------------------------------------------------------------------------------------------------------

        # Wrap model
        fmodel = KerasModel(model, bounds=(0, 1), predicts=Ptype)
        #KK: KerasModel Creates a Model instance from a Keras model.

        # Prepare attack
        #attack = foolbox.attacks.IterativeGradientSignAttack(fmodel) 
        #######attack = foolbox.attacks.DeepFoolAttack(fmodel)
        attack = foolbox.attacks.SaliencyMapAttack(fmodel)

        #attack = foolbox.attacks.BIM(fmodel)

        #attack = foolbox.attacks.LBFGSAttack(fmodel)


        # ------Get data, labels and categorical labels ***only for correctly classified examples***
        l = np.argwhere(predicted_legitimate_labels == y_test_c).shape[0]
        #this is the number of legitimate images correctly classified
        x_test_ok = np.reshape(x_test[np.array(np.argwhere(predicted_legitimate_labels == y_test_c)), :, :, :], (l, img_rows,
                                                                                                                 img_cols,
                                                                                                                img_chans))
        #put the correctly classified images in a Numpy array x_test_ok
        y_test_ok = np.reshape(y_test[np.argwhere(predicted_legitimate_labels == y_test_c), :], (l, num_classes))
        y_test_c_ok = np.argmax(y_test_ok, axis=1)

        # ------------------


        # Elaborate n_test adversarial examples ***only for correctly classified examples*** (at most l)
        n_test = l#150 # it must be lower than l
        #how many many images out of the correctly classified you want to try to attack!

        S = 0
        S_jpg  = 0
        avg_Max_dist = 0
        avg_L1_dist = 0
        avg_No_Mod_Pixels = 0
        t = 0
        avg_psnr = 0
        PSNR = 0
        psnr_org = 0



        adv_images = np.zeros((n_test, img_rows, img_cols, img_chans))
        true_labels_cat = []
        for idx in np.arange(n_test):
            image = x_test_ok[idx]

            true_labels_cat.append(y_test_ok[idx, :])

            image = image.astype('float32')

            if compressJPEG:
                img1 = Image.fromarray(np.uint8(255*image[:,:,0]))
                img1.save('temp.jpeg', "JPEG", quality=jpeg_quality)
                img_reread = Image.open('temp.jpeg')
                image = np.array(img_reread)
                image = np.reshape(image, (img_rows, img_cols, img_chans))


            # Generate adversarial images
            adv_images[idx] = attack(image, y_test_c_ok[idx])


            adversarial_image = 255 * adv_images[idx].reshape((img_rows, img_cols))


            #######################################
            #np.save('.................................' % idx,adversarial_image)
            #path_adv_Image = '..................................'
            #adversarial = adversarial_image
            #cv2.imwrite(path_adv_Image + 'adv_%d.png' % idx, adversarial)


            # Scores of legitimate and adversarial images for each idx
            scoreTemp = fmodel.predictions(image)
            true_score = foolbox.utils.softmax(scoreTemp)
            true_class = np.argmax(true_score)
            adv_score = foolbox.utils.softmax(fmodel.predictions(adv_images[idx]))
            adv_class = np.argmax(adv_score)

            print('Image {}. Class changed from {} to {}. The score passes from {} to {}'.format(idx, true_class,
                                                                                                 adv_class, true_score,
                                                                                                 adv_score))

            '''print('After rounding. Class changed from {} to {}. The score passes from {} to {}'.format(idx, true_class,
                                                                                                 Z_class, true_score,
                                                                                                 Z_score))
																								'''


            # the if below is to solve the strange problem with the prediction of a matrix of nan values...
            if np.any(np.isnan(adv_images[idx])):
                adv_class = true_class #attack not successful
                t = t + 1
                print('An adversarial image cannot be found!!')


            if true_class == adv_class:
                S = S+1

            # plot image, adv_image and difference
            #Measure the distortion between the original image and attacked image
            image_before = 255 * image.reshape((img_rows, img_cols))
            diff = np.double(image_before) - np.double(adversarial_image)
            #diff = np.double(image_before) - np.double(Z)
            print('Max distortion adversarial [After Rounding] = {:3.4f}; L1 distortion = {:3.4f}'.format(abs(diff).max(),
                                                                                                 abs(diff).sum() / (
                                                                                                             img_rows * img_cols)))
            print('Percentage of modified pixels [After Rounding]  = {:3.4f}'.format(np.count_nonzero(diff)/(img_rows * img_cols)))

            psnr_org = psnr(image_before, adversarial_image)
            print('PSNR = {:3.4f}'.format(abs(psnr_org)))

            X = np.uint8(image_before)
            #Z = np.uint8(np.round(adversarial_image))  # Omit This Line Code
            #show_figures(X,Z,true_score,Z_score)

            # to save the result of the attack, save the Z matrix.......
            #Z.save(...)

            # update average distortion
            if true_class != adv_class:
              avg_Max_dist = avg_Max_dist + abs(diff).max()
              avg_L1_dist = avg_L1_dist + abs(diff).sum()/(img_rows * img_cols)
              avg_No_Mod_Pixels = avg_No_Mod_Pixels + np.count_nonzero(diff) / (img_rows * img_cols)
              avg_psnr = avg_psnr + psnr(image_before, adversarial_image)

            # -------------------------------
            # #Compress JPEG the image and test again
            # -------------------------------

            '''if jpeg:
                #cv2.imwrite('tmp.jpg', Z[::-1], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                #adv_reread = imread('tmp.jpg')
                img1 = Image.fromarray(Z)
                img1.save('temp.jpeg', "JPEG", quality= jpeg_quality)
                adv_reread = Image.open('temp.jpeg')
                x_test_comp = np.array(adv_reread)
                x_test_comp = x_test_comp.reshape(img_rows, img_cols, img_chans)
                x_test_comp = x_test_comp.astype('float32')
                x_test_comp /= 255
                adv_reread_score = foolbox.utils.softmax(fmodel.predictions(x_test_comp))
                adv_reread_class = np.argmax(adv_reread_score)
                if true_class == adv_reread_class:
                    S_jpg = S_jpg + 1
                print('Class after JPEG compression {}, with score {}.'.format(adv_reread_class,adv_reread_score))

                x_test_comp = 255* x_test_comp.reshape((img_rows, img_cols))
			'''


        n=n_test-S
        print('Adversarial failures: {} over {}'.format(S,n_test))
        print('Average distortion: max dist {}, L1 dist {}'.format(avg_Max_dist/n,avg_L1_dist/n))
        print('Average no of modified pixels: {}'.format(avg_No_Mod_Pixels/n))
        print('The adversarial image cannot be found  {} times over {}'.format(t,n_test))


        if jpeg:
           print('Percentage of adversarial JPEG unchanged with QF {} (the attack is not successful): {}'.format(jpeg_quality, S_jpg/n_test))


        # Evaluate accuracy
        true_labels_cat = np.array(true_labels_cat)
        adv_score = model.evaluate(adv_images, true_labels_cat, verbose=0)
        #Z_score = model.evaluate(Z, true_labels_cat, verbose=0)

        score_perfect = model.evaluate(x_test_ok, y_test_ok, verbose=0)


        print('Accuracy on legitimate images (all): {:3.4f}'.format(score[1]))
        print('Accuracy on legitimate images (only correctly classified, obviously 1): {:3.4f}'.format(score_perfect[1]))
        print('Accuracy on adversarial images: {:3.4f}'.format(adv_score[1]))
        print('Attack success rate on adversarial images N1: {:3.4f}'.format(1 - adv_score[1]))
        print('Average PSNR =: {:3.4f}'.format(avg_psnr / n))
        #print('Accuracy on legitimate images (all) by mismatched model: {:3.4f}'.format(score2[1]))


        # SECOND PART
        # Load the second model and test the adversarial images


        # Label
        label3 = 1# it may be different from label because of the differences in the model.

        # Labels
        y_test_c = np.tile(label3, n_test)

        # Convert labels to one-hot with Keras
        y_test2 = keras.utils.to_categorical(y_test_c, num_classes)

        # Test
       # adv_score_mismatch = model2.evaluate(adv_images, y_test2, verbose=0)

       # print('Accuracy on adversarial images with the mismatched model: {:3.4f}'.format(adv_score_mismatch[1]))


    # Force code to run on CPU so that it does not bother concurrent tasks that use GPU(s)
    #with tf.device('/cpu:0'):  # ('/cpu:0', '/gpu:0', '/gpu:2'): # ('/cpu:0'):
    main()
