from keras.models import load_model
import numpy as np
import os
from random_training import prepare_random_data_without_spliting

def get_adv_evaluation_result_with_pattern(model, full_D_adv_features,  pattern = None, true_label=(0, 1)):
    # these are manupalted iamge prentede to be real, should be taken as manuplate, for ICIPnet_Clahe, (1,0) is real and (0,1) is false
    pattern.sort()

    subset_adv_features = full_D_adv_features[:,  pattern]

    if true_label ==(1, 0):
        subset_adv_labels = np.vstack((np.ones(full_D_adv_features.shape[0]), np.zeros(full_D_adv_features.shape[0]))).T
    elif true_label == (0,1):
        subset_adv_labels = np.vstack((np.zeros(full_D_adv_features.shape[0]), np.ones(full_D_adv_features.shape[0]))).T
    return model.evaluate(x=subset_adv_features, y=subset_adv_labels, verbose=0)

def get_prediction_result_with_pattern(model, full_D_features, pattern = None, true_label=(0, 1)):
    # these are manupalted iamge prentede to be real, should be taken as manuplate, for ICIPnet_Clahe, (1,0) is real and (0,1) is false
    pattern.sort()
    subset_adv_features = full_D_features[:, pattern]
    return model.predict_on_batch(x=subset_adv_features)


if __name__ == '__main__':
    adversarialFolder = r'D:\Random_Defense\ICIPnet_Resize\Features\advs has PSNR mis match problem'
    modelsFolder = r'D:\Random_Defense\ICIPnet_Resize\RandomTrained_results\models\features3200'
    randomPatternFolder = None#r'D:\Random_Defense\StammNet_Resize\Pattern600'
    numSubset = 3200
    fullDSize = 3200 # StammNet is 1728, ICIPnet is 3200

    # all models in this folder shares same pattern
    if randomPatternFolder is None:
        pattern = np.random.choice(fullDSize, size=numSubset, replace=False)
    else:
        patternNameList = os.listdir(randomPatternFolder)

    advEvaluationDict={}
    advPredictionDict = {}
    for advName in os.listdir(adversarialFolder):
        advEvaluationDict[advName] = []
        advPredictionDict[advName] = []


    # adv99=[]
    # adv96=[]
    # adv99fullfeature = np.load(r'D:\Random_Defense\ICIPnet_CLAHE\Features\newBIM\BIMBS_TrueEps0.02Stpsiz0.0025iters15-99.6%44.41.npy')
    # adv96fullfeature = np.load(r'D:\Random_Defense\ICIPnet_CLAHE\Features\newBIM\BIMBS_TrueEps0.005Stpsiz0.0025-96.0%51.08.npy')
    # loop on models first, one model needs evaluate on several advs
    # for modelIdx, modelName in enumerate(os.listdir(modelsFolder)):
    for i in range(50):
        if i<-1:
            continue
        # modelName = str(i)+'random_features'+str(numSubset)+'-StammNetMeadian2FC_layer-4096hidden_nodes-LR0.0001-50epochs-500batchsize-seed19930119-early_stop.h5'
                                      # 11random_features1728-StammNetMeadian2FC_layer-4096hidden_nodes-LR0.0001-50epochs-500batchsize-seed19930119-early_stop

        modelName = str(i)+'random_features' + str(numSubset)+'-ICIPnetResize1FC_layer-250hidden_nodes-LR1e-05-50epochs-100batchsize-seed19930119-early_stop.h5'

        try:
            patternName = patternNameList[i]
        except NameError:
            patternName = 'full D no pattern'
        print('dealing with #{} model: {}'.format(i, modelName))
        print('dealing with #{} pattern: {}'.format(i, patternName))
        try:
            model=load_model(os.path.join(modelsFolder,modelName))
        except OSError:
            print(OSError)
            continue
        # loop on adversarial examples
        for idx, advName in enumerate(os.listdir(adversarialFolder)):
            fullDadvFeature=np.load(os.path.join(adversarialFolder, advName))
            if randomPatternFolder is None:
                pattern = np.random.choice(fullDSize, size=numSubset, replace=False)
            else:
                pattern= np.load(os.path.join(randomPatternFolder, patternName))
            print('pattern used(fist two value of lenth {} ): {}'.format(len(pattern),pattern[:2]))
            (loss, acc)= get_adv_evaluation_result_with_pattern(model, fullDadvFeature, pattern)
            allPredictValue = get_prediction_result_with_pattern(model, fullDadvFeature, pattern)
            avePredictValueOnAllSample = np.mean(allPredictValue, axis=0)
            # every model have a full-adv dict update, in total 50*#adv updates
            advEvaluationDict[advName].append((loss, acc))
            advPredictionDict[advName].append(avePredictValueOnAllSample)

        for key, value in advEvaluationDict.items():
            print('For subset {}, mean performance of adv {} over {} patterns/tests is {}'.format(numSubset,
                                                                                                 len(value),
                                                                                                 key,
                                                                                                 np.mean(value, axis=0)))
        for key, value in advPredictionDict.items():
            print('For subset {}, mean prediction value of adv {} over {} patterns/tests is {}'
                  .format(numSubset, key, len(value), np.mean(value, axis=0)))
        del model
        # if randomPatternFolder is None:
        #     pattern = np.random.choice(fullDSize, size=numSubset, replace=False)
        # else:
        #     pattern = np.load(os.path.join(randomPatternFolder, patternName))

    # for key, value in advEvaluationDict.items():
    #     print('For subset {}, mean performance of {} adv {} only on vaild part is {}'.format(numSubset, len(value), key, np.mean(value,axis=0)))
    #     adv96.append(get_adv_evaluation_result_with_pattern(model, adv96fullfeature, pattern))
    #     adv99.append(get_adv_evaluation_result_with_pattern(model, adv99fullfeature, pattern))
    # print('adv96', np.mean(adv96,axis=0))
    # print('adv99', np.mean(adv99,axis=0))


        # pattern = np.load(os.path.join(randomPatternFolder, patternNameList[modelIdx]))
        # _, _, _, _, testing_set, testing_labels=prepare_random_data_without_spliting(
        #     numSubset,
        #     r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Pristine_Train.npy',
        #     r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Clahe_Train.npy',
        #     r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Pristine_Val.npy',
        #     r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Clahe_Val.npy',
        #     r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Pristine_Test.npy',
        #     r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Clahe_Test.npy',
        #     seed=None,
        #     FULL_FEATURE_SIZE=fullDSize,
        #     pattern=patter

        # for i in range(50):
        #     modelName = str(i) + 'random_features' + str(
        #         numSubset) + '-ICIPnetClahe1FC_layer-250hidden_nodes-LR1e-05-50epochs-100batchsize-seed19930119-early_stop.h5'
        #     try:
        #         patternName = patternNameList[i]
        #     except NameError:
        #         patternName = 'full D no pattern'
        #     print('dealing with #{} model: {}'.format(i, modelName))
        #     print('dealing with #{} pattern: {}'.format(i, patternName))
        #     model = load_model(os.path.join(modelsFolder, modelName))
        #     # loop on adversarial examples
        #     for idx, advName in enumerate(os.listdir(adversarialFolder)):
        #         fullDadvFeature = np.load(os.path.join(adversarialFolder, advName))
        #         if randomPatternFolder is None:
        #             pattern = np.random.choice(fullDSize, size=numSubset, replace=False)
        #         else:
        #             pattern = np.load(os.path.join(randomPatternFolder, patternName))
        #         print('pattern used(fist two value of lenth {} ): {}'.format(len(pattern), pattern[:2]))
        #         (loss, acc) = get_adv_evaluation_result_with_pattern(model, fullDadvFeature, pattern)
        #         # every model have a full-adv dict update, in total 50*#adv updates
        #         advEvaluationDict[advName].append((loss, acc))




