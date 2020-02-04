from keras.models import load_model,Model
from glob import glob
from keras.preprocessing.image import img_to_array, load_img
import numpy as np


if __name__ == '__main__':

    idx1 = [50,200,400,600,800,1000,1200,1500]
    sizeII = 50  # different cases (example 20 different cases)

    StammNet_model_path = r'F:\5\AttackImages\Model_original_vs_resize_RAISE8K_40epoch.h5'
    StammNet_model = load_model(StammNet_model_path)
    StammNet_model_fe = Model(inputs=StammNet_model.input, outputs=StammNet_model.get_layer('flatten_1').output)  # takes bounds[0,1] shape(1,128,128,1) as input
    Model.summary(StammNet_model_fe)

    ''' image lists '''
    StammNet_imlist_pristine_Tr = glob(r'......................\*.png')
    StammNet_imlist_pristine_Val = glob(r'......................\*.png')
    StammNet_imlist_pristine_Te = glob(r'......................\*.png')

    StammNet_imlist_resize08_Tr = glob(r'......................\*.png')
    StammNet_imlist_resize08_Val = glob(r'......................\*.png')
    StammNet_imlist_resize08_Te = glob(r'......................\*.png')

    StammNet_imlist_JSMA01 = glob(r'...............\*.npy')


    StammNet_imlist_IFGSM10 = glob(r'...............\*.npy')
 


    ###########################################################
    #Pristine TR + Val + Test
    ###########################################################

    feStamm_pristine_Tr = np.zeros(1728)
    for idx, im_file in enumerate(StammNet_imlist_pristine_Tr):
        print('dealing with number {} image in StammNet_imlist_pristine_Tr'.format(idx))
        image = img_to_array(load_img(im_file, target_size=(64, 64), color_mode='grayscale')) / 255
        feStamm_pristine_Tr = np.vstack((feStamm_pristine_Tr, StammNet_model_fe.predict(image.reshape(1, 64, 64, 1))))
    feStamm_pristine_Tr = np.delete(feStamm_pristine_Tr, 0, 0)
    np.save('F:/5/test/inter_features/StammNet_model_fe_pristine_Tr.npy', feStamm_pristine_Tr)

    feStamm_pristine_Val = np.zeros(1728)
    for idx, im_file in enumerate(StammNet_imlist_pristine_Val):
        print('dealing with number {} image in StammNet_imlist_pristine_Val'.format(idx))
        image = img_to_array(load_img(im_file, target_size=(64, 64), color_mode='grayscale')) / 255
        feStamm_pristine_Val = np.vstack((feStamm_pristine_Val, StammNet_model_fe.predict(image.reshape(1, 64, 64, 1))))
    feStamm_pristine_Val = np.delete(feStamm_pristine_Val, 0, 0)
    np.save('F:/5/test/inter_features/StammNet_model_fe_pristine_Val.npy', feStamm_pristine_Val)

    feStamm_pristine_Te = np.zeros(1728)
    for idx, im_file in enumerate(StammNet_imlist_pristine_Te):
        print('dealing with number {} image in StammNet_imlist_pristine_Te'.format(idx))
        image = img_to_array(load_img(im_file, target_size=(64, 64), color_mode='grayscale')) / 255
        feStamm_pristine_Te = np.vstack((feStamm_pristine_Te, StammNet_model_fe.predict(image.reshape(1, 64, 64, 1))))
    feStamm_pristine_Te = np.delete(feStamm_pristine_Te, 0, 0)
    np.save('F:/5/test/inter_features/StammNet_model_fe_pristine_Te.npy', feStamm_pristine_Te)

    concatenate_Pristine = np.concatenate((feStamm_pristine_Tr[0:100000, :], feStamm_pristine_Val[0:3000, :],feStamm_pristine_Te[0:10000, :]), axis=0)
    np.save('F:/5/test/inter_features/StammNet_model_fe_pristine_Concat.npy', concatenate_Pristine)

    #########################################################################
    #      Resize08 flatten features  (Tr+Val+Te)
    #########################################################################
    feStamm_resize08_Tr = np.zeros(1728)
    for idx, im_file in enumerate(StammNet_imlist_resize08_Tr):
        print('dealing with number {} image in StammNet_imlist_resize08_Tr'.format(idx))
        image = img_to_array(load_img(im_file, target_size=(64, 64), color_mode='grayscale')) / 255
        feStamm_resize08_Tr = np.vstack((feStamm_resize08_Tr, StammNet_model_fe.predict(image.reshape(1, 64, 64, 1))))
    feStamm_resize08_Tr = np.delete(feStamm_resize08_Tr, 0, 0)
    np.save('F:/5/test/inter_features/StammNet_model_fe_resize08_Tr.npy', feStamm_resize08_Tr)

    feStamm_resize08_Val = np.zeros(1728)
    for idx, im_file in enumerate(StammNet_imlist_resize08_Val):
        print('dealing with number {} image in StammNet_imlist_resize08_Val'.format(idx))
        image = img_to_array(load_img(im_file, target_size=(64, 64), color_mode='grayscale')) / 255
        feStamm_resize08_Val = np.vstack((feStamm_resize08_Val, StammNet_model_fe.predict(image.reshape(1, 64, 64, 1))))
    feStamm_resize08_Val = np.delete(feStamm_resize08_Val, 0, 0)
    np.save('F:/5/test/inter_features/StammNet_model_fe_resize08_Val.npy', feStamm_resize08_Val)

    feStamm_resize08_Te = np.zeros(1728)
    for idx, im_file in enumerate(StammNet_imlist_resize08_Te):
        print('dealing with number {} image in StammNet_imlist_resize08_Te'.format(idx))
        image = img_to_array(load_img(im_file, target_size=(64, 64), color_mode='grayscale')) / 255
        feStamm_resize08_Te = np.vstack((feStamm_resize08_Te, StammNet_model_fe.predict(image.reshape(1, 64, 64, 1))))
    feStamm_resize08_Te = np.delete(feStamm_resize08_Te, 0, 0)
    np.save('F:/5/test/inter_features/StammNet_model_fe_resize08_Te.npy', feStamm_resize08_Te)

    concatenate_resize08 = np.concatenate((feStamm_resize08_Tr[0:100000, :], feStamm_resize08_Val[0:3000, :], feStamm_resize08_Te[0:10000, :]), axis=0)
    np.save('F:/5/test/inter_features/StammNet_model_fe_resize08_Concat.npy', concatenate_resize08)


    ########################################################################
    #      Adversarial JSMA 0.01
    ########################################################################
    feStamm_JSMA001 = np.zeros(1728)
    for idx, im_file in enumerate(StammNet_imlist_JSMA001):
        print('dealing with number {} image in StammNet_imlist_JSMA001'.format(idx))
        image = np.load(im_file)
        image = image / 255
        feStamm_JSMA001 = np.vstack((feStamm_JSMA001, StammNet_model_fe.predict(image.reshape(1, 64, 64, 1))))
    feStamm_JSMA001 = np.delete(feStamm_JSMA001, 0, 0)
    np.save('F:/5/test/inter_features/StammNet_model_fe_JSMA001.npy', feStamm_JSMA001)

    ########################################################################
    #      IFGSM eps 10 
    ########################################################################
    feStamm_IFGSM10 = np.zeros(1728)
    for idx, im_file in enumerate(StammNet_imlist_IFGSM10):
        print('dealing with number {} image in StammNet_imlist_IFGSM10'.format(idx))
        image = np.load(im_file)
        image = image / 255
        feStamm_IFGSM10 = np.vstack((feStamm_IFGSM10, StammNet_model_fe.predict(image.reshape(1, 64, 64, 1))))
    feStamm_IFGSM10 = np.delete(feStamm_IFGSM10, 0, 0)
    np.save('F:/5/test/inter_features/StammNet_model_fe_IFGSM10.npy', feStamm_IFGSM10)

  

    ########################################################################
    #     Load all features
    ########################################################################

    pristine = np.load('F:/5/test/inter_features/StammNet_model_fe_pristine_Concat.npy')
    resize08 = np.load('F:/5/test/inter_features/StammNet_model_fe_resize08_Concat.npy')
    JSMA01 = np.load('F:/5/test/inter_features/StammNet_model_fe_JSMA01.npy')
    IFGSM10 = np.load('F:/5/test/inter_features/StammNet_model_fe_IFGSM10.npy')
   

    print('pristine shape', pristine.shape)
    print('resize08 shape', resize08.shape)
    print('JSMA01 shape', JSMA01.shape)
    print('IFGSM10 shape', IFGSM10.shape)


    assert len(pristine[0]) == pristine.shape[1]
    index = np.array(range(pristine.shape[1]))


    for I in idx1:
     for idx in np.arange(sizeII):
        Stamm_random_pattern = np.random.choice(index, size=I, replace=False)
        Stamm_random_pattern.sort()
        np.save('F:/5/test/inter_features_{}/StammNet_random_pattern_{}.npy'.format(I,idx), Stamm_random_pattern)

        '''Using pattern generate subset'''
        pattern_path = ('F:/5/test/inter_features_{}/StammNet_random_pattern_{}.npy'.format(I,idx))
        Stamm_random_pattern = np.load(pattern_path)


        Stamm_pristine = np.array([pristine[:, j] for j in Stamm_random_pattern]).T
        Stamm_resize08 = np.array([resize08[:, j] for j in Stamm_random_pattern]).T

        Stamm_JSMA01 = np.array([JSMA01[:, j] for j in Stamm_random_pattern]).T

        Stamm_IFGSM10 = np.array([IFGSM10[:, j] for j in Stamm_random_pattern]).T


        np.save('F:/5/test/inter_features_{}/StammNet_subset_pristine_{}.npy'.format(I,idx), Stamm_pristine)
        np.save('F:/5/test/inter_features_{}/StammNet_subset_resize08_{}.npy'.format(I, idx), Stamm_resize08)

        np.save('F:/5/test/inter_features_{}/StammNet_subset_JSMA01_{}.npy'.format(I, idx), Stamm_JSMA01)

        np.save('F:/5/test/inter_features_{}/StammNet_subset_IFGSM10_{}.npy'.format(I, idx), Stamm_IFGSM10)
        







