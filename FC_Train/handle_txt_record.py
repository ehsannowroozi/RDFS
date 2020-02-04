import numpy as np

def give_me_all_loss_and_acc(txt_path, identifier='No Attack'):
    # takes one txt file, go through it line by line
    # takes by default "['loss', 'acc'] =" as input, return what's behind it as loss list and acc list
    with open(txt_path,'r') as file:
        loss_record=[]
        acc_record = []
        for line in file:
            if identifier in line: # this line contains what you want
                str_results = line.split('=')[1]
                str_results = str_results.strip(' []\n')
                loss, acc = np.array(str_results.split(', '),dtype=np.float32)
                loss_record.append(loss)
                acc_record.append(acc)
    return loss_record, acc_record

if __name__ =='__main__':
    txt_path = r'.......................................................txt'
    identifier_list = ['No Attack',
                       # 'StammNet_model_fe_BIM02.npy', 'StammNet_model_fe_BIM03.npy','StammNet_model_fe_IFGSM10.npy',
                       # 'StammNet_model_fe_IFGSM100.npy', '..........................................npy',
                       # 'StammNet_model_fe_JSMA01.npy','StammNet_model_fe_LBFGS.npy',
                       'StammNet_model_fe_BIM0.3_Limit40.npy',
                       'StammNet_model_fe_BIM0.3_Limit50.npy',
                       'StammNet_model_fe_BIM0.3_Limit55.npy',
                       'StammNet_model_fe_LBFGS_Limit40.npy',
                      'StammNet_model_fe_LBFGS_Limit50.npy',
                       'StammNet_model_fe_LBFGS_Limit55.npy',
                       ]

    for identifier in identifier_list:
        losses, acces = give_me_all_loss_and_acc(txt_path, identifier)
        assert len(losses)==len(acces)
        print('Average loss={} acc={}%, of {} case, over {} runtimes'.format(np.mean(losses), np.mean(acces)*100, identifier, len(losses)))