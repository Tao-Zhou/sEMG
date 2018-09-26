'''
对测试集做预测，依次作为修改模型的标准
'''
from sEMG_classifier import sEMG_classifier
import os
import numpy as np
import json

if __name__ == '__main__':
    for model_path in ['svm1.m','linearSVC1.m','k_neighbor1.m'] :
        a = sEMG_classifier(1000,300)
        a.load_model('E://python//'+model_path)
        cv_url = 'E://srt//CapgMyoData//dbc_proc_python//cv_set'
        label_num = len(os.listdir(cv_url))
        subject_num = 10
        table = {}
        for i in range(subject_num):
                table['subject_'+str(i+1)] = {}
        
        #对每个label-subject文件夹下的文件进行评估并将结果写入
        for label in os.listdir(cv_url):
            label_idx = int(label)
            label_dir = os.path.join(cv_url,label)
            #print('label_idx:'+str(label_idx))
            for subject in os.listdir(label_dir):
                subject_idx = int(subject[-3:])
                subject_dir = os.path.join(label_dir,subject)
                print('subject_'+str(subject_idx)+' label_'+str(label_idx)+' assessing................')
                b = a.tranverse_files(subject_dir)
                try:
                    table['subject_'+str(subject_idx)][str(label_idx)] = b
                except ZeroDivisionError:
                    print('zero divide')
                else:
                    pass
                #print(subject_idx)
        
        for subject,accuracy in table.items():
            print(subject+':')
            print(accuracy)
            for key,value in accuracy.items():
                value[0] = int(value[0])
                value[1] = int(value[1])
        f = open(model_path[:-2]+'.json','w')
        json.dump(table,f)
        f.close()