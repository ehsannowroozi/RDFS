from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_training_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def prepare_random_data_with_spliting(sample_size, PRISTINE_path, MANIPULATE_path,seed, pattern = None): #
    # load data, stack, prepare for random selecting
    pristine = np.load(PRISTINE_path)
    manipulate = np.load(MANIPULATE_path)

    if pattern is None:
        # randomly selecting features using a seed
        np.random.seed(seed)
        pattern = np.random.choice(pristine.shape[1], size=sample_size, replace=False)
    pattern.sort()
    sample_pristine = pristine[:,pattern]
    sample_resize = manipulate[:,pattern]

    # spilt dataset into training validation and testing
    # pristine label is [0,1]
    # resize label is [1,0]
    num_samples = sample_pristine.shape[0]
    sp_validation, sp_train, sp_test = np.split(sample_pristine,[1000,16000])
    # sp_train, sp_validation, sp_test = np.split(sample_pristine, [int(0.6 * num_samples), int(0.8 * num_samples)])
    sp_train_labels = np.vstack((np.zeros(sp_train.shape[0]), np.ones(sp_train.shape[0]))).T
    sp_validation_labels = np.vstack((np.zeros(sp_validation.shape[0]), np.ones(sp_validation.shape[0]))).T
    sp_test_labels = np.vstack((np.zeros(sp_test.shape[0]), np.ones(sp_test.shape[0]))).T

    sr_validation, sr_train, sr_test = np.split(sample_resize, [1000, 16000])
    # sr_train, sr_validation, sr_test = np.split(sample_resize, [int(0.6 * num_samples), int(0.8 * num_samples)])
    sr_train_labels = np.vstack((np.ones(sr_train.shape[0]), np.zeros(sr_train.shape[0]))).T
    sr_validation_labels = np.vstack((np.ones(sr_validation.shape[0]), np.zeros(sr_validation.shape[0]))).T
    sr_test_labels = np.vstack((np.ones(sr_test.shape[0]), np.zeros(sr_test.shape[0]))).T

    training_set = np.vstack((sp_train,sr_train))
    training_labels = np.vstack((sp_train_labels, sr_train_labels))

    validation_set = np.vstack((sp_validation,sr_validation))
    validation_labels = np.vstack((sp_validation_labels, sr_validation_labels))

    testing_set = np.vstack((sp_test, sr_test))
    testing_labels = np.vstack((sp_test_labels, sr_test_labels))


    return training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels


def process_line(line, pattern):
    #pattern should be list or array
    line=line.split(',')
    x=np.float64(line[:-2])
    y=np.float64(line[-2:])
    x=x[pattern]
    return x,y
def prepare_random_data_generator(batch_size, dataPath, pattern):
    while 1:
        f = open(dataPath)
        cnt=0
        X=[]
        Y=[]
        for line in f:
            x,y = process_line(line,pattern)
            X.append(x)
            Y.append(y)
            cnt+=1
            if cnt == batch_size:
                cnt=0
                yield (np.array(X),np.array(Y))
                X=[]
                Y=[]
        f.close()


def prepare_random_data_without_spliting(sample_size, PRISTINE_training_path, MANIPULATE_training_path,
PRISTINE_validation_path, MANIPULATE_validation_path, PRISTINE_testing_path, MANIPULATE_testing_path, seed, FULL_FEATURE_SIZE, pattern = None):
    if pattern is None:
        # randomly selecting features using a seed
        np.random.seed(seed)
        pattern = np.random.choice(FULL_FEATURE_SIZE, size=sample_size, replace=False)
    pattern.sort()
    PRISTINE_training = np.load(PRISTINE_training_path)
    PRISTINE_training = PRISTINE_training[:, pattern]
    MANIPULATE_training = np.load(MANIPULATE_training_path)
    MANIPULATE_training = MANIPULATE_training[:, pattern]
    PRISTINE_train_labels = np.vstack((np.ones(PRISTINE_training.shape[0]), np.zeros(PRISTINE_training.shape[0]))).T
    MANIPULATE_train_labels = np.vstack((np.zeros(MANIPULATE_training.shape[0]), np.ones(MANIPULATE_training.shape[0]))).T

    training_set = np.vstack((PRISTINE_training, MANIPULATE_training))
    training_labels = np.vstack((PRISTINE_train_labels, MANIPULATE_train_labels))
    del PRISTINE_training,MANIPULATE_training,PRISTINE_train_labels,MANIPULATE_train_labels

    PRISTINE_validation = np.load(PRISTINE_validation_path)
    PRISTINE_validation = PRISTINE_validation[:, pattern]
    MANIPULATE_validation = np.load(MANIPULATE_validation_path)
    MANIPULATE_validation = MANIPULATE_validation[:, pattern]
    PRISTINE_val_labels = np.vstack((np.ones(PRISTINE_validation.shape[0]), np.zeros(PRISTINE_validation.shape[0]))).T
    MANIPULATE_val_labels = np.vstack((np.zeros(MANIPULATE_validation.shape[0]), np.ones(MANIPULATE_validation.shape[0]))).T

    validation_set = np.vstack((PRISTINE_validation,MANIPULATE_validation))
    validation_labels = np.vstack((PRISTINE_val_labels, MANIPULATE_val_labels))
    del PRISTINE_validation,MANIPULATE_validation,PRISTINE_val_labels,MANIPULATE_val_labels

    PRISTINE_testing = np.load(PRISTINE_testing_path)
    PRISTINE_testing = PRISTINE_testing[:, pattern]
    MANIPULATE_testing = np.load(MANIPULATE_testing_path)
    MANIPULATE_testing = MANIPULATE_testing[:, pattern]
    PRISTINE_test_labels = np.vstack((np.ones(PRISTINE_testing.shape[0]), np.zeros(PRISTINE_testing.shape[0]))).T
    MANIPULATE_test_labels = np.vstack((np.zeros(MANIPULATE_testing.shape[0]), np.ones(MANIPULATE_testing.shape[0]))).T

    testing_set = np.vstack((PRISTINE_testing, MANIPULATE_testing))
    testing_labels = np.vstack((PRISTINE_test_labels, MANIPULATE_test_labels))
    del PRISTINE_testing,MANIPULATE_testing,PRISTINE_test_labels,MANIPULATE_test_labels

    return training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels


def get_adv_evaluation_result_with_or_without_pattern(model, full_D_adv_features, sample_size, pattern = None):

    if pattern is not None:
        pattern.sort()
        subset_adv_features = full_D_adv_features[:,  pattern]
        subset_adv_labels = np.vstack((np.zeros(full_D_adv_features.shape[0]), np.ones(full_D_adv_features.shape[0]))).T
        return model.evaluate(x=subset_adv_features, y=subset_adv_labels, verbose=0)

    else:
        pattern = np.random.choice(full_D_adv_features.shape[1], size=sample_size, replace=False)
        pattern.sort()
        subset_adv_features = full_D_adv_features[:, pattern]
        subset_adv_labels = np.vstack((np.zeros(full_D_adv_features.shape[0]), np.ones(full_D_adv_features.shape[0]))).T
        return model.evaluate(x=subset_adv_features, y=subset_adv_labels, verbose=0)


def build_top_layer(input_size,hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, input_shape=(input_size,),activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(hidden_nodes, input_shape=(input_size,), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax',name='predictinos'))
    model.compile(Adam(lr=LEARNING_RATE),loss='binary_crossentropy', metrics=['accuracy'])
    return model

# def get_stammnet_conv_layers(model):
#     return Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)

if __name__ =='__main__':

    LEARNING_RATE = 0.00001
    NUM_FEATURES_USED = 600
    FULL_FEATURE_SIZE = 3200
    HIDDEN_NODES = 250
    RANDOM_SEED = 19930119
    EPOCHS = 50
    BATCH_SIZE = 100
    EARLY_STOP_PATIENCE = 5
    PATTERN_FOLDER = r'D:\Random_Defense\Ehsan give me7.9\ICIPnet_Resize\Pattern600'
    RUN_TIMES = 50
    RUN_TIME_continue = -1 # continue from number 23 runtime
    MODEL_SAVE_PATH = r'D:\Random_Defense\ICIPnet_CLAHE\RandomTrained_results-doublecheck\features600'
    MODEL_NAME = 'random_features{}-ICIPnetResize1FC_layer-{}hidden_nodes-LR{}-{}epochs-{}batchsize-seed{}-early_stop.h5'.format(NUM_FEATURES_USED, HIDDEN_NODES, LEARNING_RATE, EPOCHS, BATCH_SIZE, RANDOM_SEED)

    # feature dataset paths
    PRISTINE_training = r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Pristine_Train.npy'
    MANIPULATE_training=r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Clahe_Train.npy'
    PRISTINE_validation=r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Pristine_Val.npy'
    MANIPULATE_validation=r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Clahe_Val.npy'
    PRISTINE_testing=r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Pristine_Test.npy'
    MANIPULATE_testing=r'D:\Random_Defense\ICIPnet_CLAHE\Features\ICIPnet_model_fe_CLAHE_Clahe_Test.npy'

    prepare_data_function = prepare_random_data_without_spliting

    PRINSTINE_Concat=None
    MANIPULATE_Concat=None

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    ADVERSARIAL_FEATURE_FOLDER = r'D:\Random_Defense\ICIPnet_CLAHE\Features\newBIM'

    # need to build a new model that contains previous Conv layers and new top layers
    # stammnet = load_model(
    #     r'D:\Anti-Spoofing\Anti-anti-spoofing\Investigating-Adversarials\random_train_NN_StammNet\model_original_vs_resize_RAISE8K.h5')



    ''' for loop on different patterns, do both training and evaluation within one loop'''
    # some variables to compute average
    no_attack_evaluations = np.array([0,0], dtype='float64')
    adversarial_evaluations_dict={'BIMBS_TrueEps0.02Stpsiz0.0025iter15runtime2-99.6%44.41.npy':np.array([0, 0], dtype='float64'),
                             'BIMBS_TrueEps0.02Stpsiz0.0025iters15-99.6%44.41.npy':np.array([0,0], dtype='float64'),
                                  'BIMBS_TrueEps0.005Stpsiz0.0025-96.0%51.08.npy': np.array([0, 0], dtype='float64'),
                             'BIMBS_TrueEps0.005Stpsiz0.0025runtime2-96.0%51.08.npy':np.array([0,0], dtype='float64'),
                             # 'StammNet_model_fe_LBFGS_Limit50.npy':np.array([0,0], dtype='float64'),
                             #       'StammNet_model_fe_LBFGS_Limit55.npy': np.array([0, 0], dtype='float64'),
                             # 'StammNet_model_fe_IFGSM100_35db.npy':np.array([0,0], dtype='float64'),
                             # 'StammNet_model_fe_IFGSM100_40db.npy':np.array([0,0], dtype='float64'),
                             #      'StammNet_model_fe_JSMA01.npy':np.array([0,0], dtype='float64'),
                             #      'StammNet_model_fe_JSMA001.npy':np.array([0,0], dtype='float64'),
                             # #      'StammNet_model_fe_BIM02.npy': np.array([0, 0], dtype='float64'),
                             # #      'StammNet_model_fe_BIM03.npy': np.array([0, 0], dtype='float64'),
                             #     'StammNet_model_fe_LBFGS.npy': np.array([0, 0], dtype='float64')
                                  }


    if PATTERN_FOLDER is None:
        for runtime in range(RUN_TIMES):
            if runtime < RUN_TIME_continue:
                continue
            pattern = np.random.choice(FULL_FEATURE_SIZE, NUM_FEATURES_USED, replace=False)
            pattern.sort()
            # training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels = prepare_random_data(
            #     NUM_FEATURES, RANDOM_SEED, pattern)
            top_fc_layers = build_top_layer(NUM_FEATURES_USED, HIDDEN_NODES)
            training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels = prepare_data_function(
                NUM_FEATURES_USED, PRISTINE_training, MANIPULATE_training, PRISTINE_validation, MANIPULATE_validation,PRISTINE_testing,
                MANIPULATE_testing, RANDOM_SEED, FULL_FEATURE_SIZE, pattern)
            print('training set.shape:{}\nvalidation set.shape:{}\ntesting set.shape:{}'.format(training_set.shape, validation_set.shape, testing_set.shape))
            ''' start training'''
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
            history = top_fc_layers.fit(x=training_set, y=training_labels,
                                        validation_data=(validation_set, validation_labels),
                                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping],verbose=2)
            top_fc_layers.save(os.path.join(MODEL_SAVE_PATH,str(runtime)+MODEL_NAME))
            """ Evaluation on no attack featres"""
            evaluation = top_fc_layers.evaluate(x=testing_set, y=testing_labels)
            no_attack_evaluations += np.array(evaluation)
            print(top_fc_layers.metrics_names, '=', evaluation)
            with open(os.path.join(MODEL_SAVE_PATH, MODEL_NAME.replace('.h5', '.txt')), 'a') as file:
                file.write('Subset {}:Model\'s performance on Test set(No Attack), runtime {}: {} = {}\n'.format(NUM_FEATURES_USED, runtime, str(
                    top_fc_layers.metrics_names), str(evaluation)))

            ''' test trained FC-layers with adversarial sample's features(resize images pretend to be pristine'''
            # adversarial_features_folder = r'D:\Anti-Spoofing\Anti-anti-spoofing\Investigating-Adversarials\random_train_NN_StammNet\ReportRandomzation(3)\adversarial features'

            # for loop on adversarial features obtained from different kinds of attacks, i.e. one row in Ehsan's table
            # don't need to calculate any average on different kinds of attacks
            for attack_filename in os.listdir(ADVERSARIAL_FEATURE_FOLDER):
                full_D_adv_features = np.load(os.path.join(ADVERSARIAL_FEATURE_FOLDER, attack_filename))
                # advs are resize image pretend to be pristine, so label should be [1, 0]
                adv_labels = np.vstack(
                    (np.ones(full_D_adv_features.shape[0]), np.zeros(full_D_adv_features.shape[0]))).T

                # when subset of features used
                # select subset randomly using pattern(if had)
                # one pattern needed( if no preivious pattern), the average shall be calculate on initial loop
                # need a loop on different patterns, however this is already done outside the loop of different attacks, so for every attack, should be
                # a way to average on different patterns
                evaluation_results = get_adv_evaluation_result_with_or_without_pattern(top_fc_layers,
                                                                                       full_D_adv_features,
                                                                                       sample_size=NUM_FEATURES_USED,
                                                                                       pattern=pattern)
                with open(os.path.join(MODEL_SAVE_PATH, MODEL_NAME.replace('.h5', '.txt')), 'a') as file:
                    file.write(
                        'Subset {} Model\'s performance on adv set{}, runtime {}: {} = {}\n'.format(NUM_FEATURES_USED, attack_filename, runtime, str(
                            top_fc_layers.metrics_names), str(evaluation_results)))
                # dict adversarial_evaluations been defined before, keys are adv npy names, values are [0,0] arrays
                adversarial_evaluations_dict[attack_filename] += np.array(evaluation_results)
                del full_D_adv_features
            del top_fc_layers
            del training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels

        print('Subset{}: Average evaluation on no attack features over {} runtimes:{}'.format(NUM_FEATURES_USED, RUN_TIMES, no_attack_evaluations / RUN_TIMES))
        with open(os.path.join(MODEL_SAVE_PATH, MODEL_NAME.replace('.h5', '.txt')), 'a') as file:
            file.write('Subset{}: AVERAGE evaluation on No Attack features:{}'.format(NUM_FEATURES_USED,
                                                                                      no_attack_evaluations / RUN_TIMES))
        for (keys, values) in adversarial_evaluations_dict.items():
            print('Subset{}: AVERAGE loss, ACC of adv {} over {} runtimes: {}'.format(NUM_FEATURES_USED, keys, RUN_TIMES, values / RUN_TIMES))
            with open(os.path.join(MODEL_SAVE_PATH, MODEL_NAME.replace('.h5', '.txt')), 'a') as file:
                file.write('Subset{}: AVERAGE evaluation on No Attack features over{} runtimes:{}'.format(NUM_FEATURES_USED, RUN_TIMES,
                                                                                                          no_attack_evaluations / RUN_TIMES))
    else:
        for runtime, patternfile in enumerate(os.listdir(PATTERN_FOLDER)):
            if runtime < RUN_TIME_continue:
                continue
            print('current pattern filename:{}'.format(os.path.join(PATTERN_FOLDER,patternfile)))
            pattern = np.load(os.path.join(PATTERN_FOLDER, patternfile))
            pattern.sort()
            # training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels = prepare_random_data(NUM_FEATURES, RANDOM_SEED, pattern)
            training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels = prepare_data_function(
                NUM_FEATURES_USED,PRISTINE_training,MANIPULATE_training,PRISTINE_validation,MANIPULATE_validation,PRISTINE_testing,MANIPULATE_testing, RANDOM_SEED,FULL_FEATURE_SIZE, pattern)
            top_fc_layers = build_top_layer(NUM_FEATURES_USED, HIDDEN_NODES)
            ''' start training'''
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
            history = top_fc_layers.fit(x=training_set, y=training_labels, validation_data=(validation_set, validation_labels),
                                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping],verbose=2)
            top_fc_layers.save(os.path.join(MODEL_SAVE_PATH, str(runtime)+MODEL_NAME))
            # visualize_training_history(history)
            # top_fc_layers.save(os.path.join(model_save_path, model_name))

            """ Evaluation on no attack featres"""
            evaluation = top_fc_layers.evaluate(x=testing_set, y=testing_labels)
            no_attack_evaluations+=np.array(evaluation)
            print(top_fc_layers.metrics_names, '=', evaluation)
            with open(os.path.join(MODEL_SAVE_PATH, MODEL_NAME.replace('.h5', '.txt')), 'a') as file:
                file.write('Subset{}: Model\'s performance on Test set(No Attack), runtime {}: {} = {}\n'.format(NUM_FEATURES_USED, runtime, str(top_fc_layers.metrics_names), str(evaluation)))

            ''' test trained FC-layers with adversarial sample's features(resize images pretend to be pristine'''


            # for loop on adversarial features obtained from different kinds of attacks, i.e. one row in Ehsan's table
            # don't need to calculate any average on different kinds of attacks
            for attack_filename in os.listdir(ADVERSARIAL_FEATURE_FOLDER):
                full_D_adv_features = np.load(os.path.join(ADVERSARIAL_FEATURE_FOLDER, attack_filename))
                # advs are resize image pretend to be pristine, so label should be [1, 0]
                adv_labels = np.vstack((np.ones(full_D_adv_features.shape[0]), np.zeros(full_D_adv_features.shape[0]))).T

                # when subset of features used
                # select subset randomly using pattern(if had)
                # one pattern needed( if no preivious pattern), the average shall be calculate on initial loop
                # need a loop on different patterns, however this is already done outside the loop of different attacks, so for every attack, should be
                # a way to average on different patterns
                evaluation_results = get_adv_evaluation_result_with_or_without_pattern(top_fc_layers, full_D_adv_features, sample_size=NUM_FEATURES_USED, pattern=pattern)
                with open(os.path.join(MODEL_SAVE_PATH, MODEL_NAME.replace('.h5', '.txt')), 'a') as file:
                    file.write('Subset{}: Model\'s performance on adv set{}, runtime {}: {} = {}\n'.format(NUM_FEATURES_USED, attack_filename, runtime, str(
                        top_fc_layers.metrics_names), str(evaluation_results)))
                # dict adversarial_evaluations been defined before, keys are adv npy names, values are [0,0] arrays
                adversarial_evaluations_dict[attack_filename] += np.array(evaluation_results)
                del full_D_adv_features
            del top_fc_layers
            del training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels


        print('Subset{}: Average evaluation on NO ATTACK features over{} runtimes:{}\n'.format(NUM_FEATURES_USED, RUN_TIMES, no_attack_evaluations / RUN_TIMES))
        with open(os.path.join(MODEL_SAVE_PATH, MODEL_NAME.replace('.h5', '.txt')), 'a') as file:
            file.write('Subset{}: AVERAGE evaluation on No Attack features over{} runtimes:{}\n'.format(NUM_FEATURES_USED, RUN_TIMES, no_attack_evaluations / RUN_TIMES))
        for (keys, values) in adversarial_evaluations_dict.items():
            print('Subset{}: AVERAGE loss, ACC of adv {} over {} runtimes: {}\n'.format(NUM_FEATURES_USED, keys, RUN_TIMES, values / RUN_TIMES))
            with open(os.path.join(MODEL_SAVE_PATH, MODEL_NAME.replace('.h5', '.txt')), 'a') as file:
                file.write('Subset{}: AVERAGE loss, ACC of adv {} over {} runtimes: {}\n'.format(NUM_FEATURES_USED, keys, RUN_TIMES, values / RUN_TIMES))



