'''
process the s_EMG data 
divide the data to train_set,cross_validation_set,test_set
all raw semg data are stored in path
'''
import os
import scipy.io as sio
import numpy as np
from matplotlib.pyplot import *
import shutil

class data_proc(object):
    

    def __init__(self,path,store_path):
        self.__path = path      
        self.__store_path = store_path

    def data_process(self):
        '''
        save the semg data in the format:
        subject_label_num.mat
        one mat file contain only one array correspond to one label 
        '''
        
        files_dir = os.listdir(self.__path)
        #print(files_dir)
        for dir in files_dir:
            files = os.listdir(self.__path+'//'+dir)
            
            for file in files:
                s_EMG = sio.loadmat(self.__path+'//'+dir+'//'+file)
                s_EMG_proc = self.data_spilt(s_EMG)  
                dir_name = self.__store_path+'//'
                print(file)
               

                for s_l in s_EMG_proc:
                    label = s_l['label']
                    if not os.path.exists(dir_name + str(label) ):
                        os.makedirs(dir_name + str(label) )
                    if np.shape(s_l['data'])[1]==0:
                        print('error!')
                    else:
                        #print(np.shape(s_l['data']))
                        sio.savemat(dir_name+str(label)+ '\\'+file[:-8]+'_'+str(label)+'_'+ \
                        str(len(os.listdir(dir_name+str(label))))+'.mat',s_l)
                    #file name format is 'subject_label_num.mat'
                    
                
                
            
    def data_spilt(self,semg_data):
        '''
        spilt the array with label ！= 0 
        '''
        data = semg_data['data'].transpose()
        label = semg_data['gesture'][0]
        #print(label)
        x = np.diff(np.insert(label,0,0))   
        a0 = np.where(x==np.max(x))[0]
        a1 = np.where(x==np.min(x))[0]
        if(len(a0) > len(a1)):
            a1.append(np.shape(label)[0])
        else:
            pass

        a0 = np.array(a0)
        a1 = np.array(a1)
        semg_label = []
       
        for idx in range(len(a0)):
            semg_label.append({'data':data[:,a0[idx]:a1[idx]],'label':np.max(x)})

        if a0[0] > 0:
            a1 = np.insert(a1,0,0)
        else:
            a0 = np.delete(a0,0)
        
        if a1[-1] >= np.shape(label)[0]:
            a1 = np.delete(a1,np.shape(a1)[0])
        else:
            a0 = np.insert(a0,np.shape(a0)[0],np.shape(label)[0])
        for idx in range(len(a0)):
            semg_label.append({'data':data[:,a1[idx]:a0[idx]],'label':0})
        
        #delete data with length 0
        a = [x for x in semg_label if np.shape(x['data'])[1] > 0]
        semg_label = a

        if len(semg_label) < 10:
            print('data sets num not enough')
        def spilt_data(one_dict):
            t = np.shape(one_dict['data'])[1]
            a = {'data':one_dict['data'][:,:t],'label':one_dict['label']}
            b = {'data':one_dict['data'][:,t:],'label':one_dict['label']}
            return (a,b)
        
        #将原有数据段再进行剪切，使得至少有10组数据
        while len(semg_label) < 10:
            semg_label = sorted(semg_label,key=lambda x: np.shape(x['data'])[0])
            a = semg_label[:-1]
            x = spilt_data(semg_label[-1])
            a.append(x[0])
            a.append(x[1])
            semg_label = a
            print(len(semg_label))

        return semg_label
        
    def divide_data(self,data_set_path):
        #divide all data into train,cross_validation,test sets
        #path includes all data
        import random

        dirs = os.listdir(data_set_path)
        #print(dirs)
        for dir in dirs:
            files = list(os.walk(data_set_path+'//'+dir))[0][2]
            #print(files)

            subject_dict = {}
            for file in files:
                subject = file[:3]
                if subject in subject_dict.keys():
                    subject_dict[subject].append(file)
                else:
                    subject_dict[subject] = [file]
            
            for subject,file in subject_dict.items():

                random.shuffle(file)
                l = len(file)
                t1 = int(l*0.6)
                t2 = int(l*0.8)
                
                #print(t1)
                train_set = file[:t1]
                cv_set = file[t1:t2]
                test_set = file[t2:]
                self.move_file(train_set,data_set_path+'//'+dir,data_set_path+'//train_set'+'//'+dir+ '//'+'subject_'+ subject)
                self.move_file(cv_set,data_set_path+'//'+dir,data_set_path+'//cv_set'+'//'+dir+ '//'+'subject_'+ subject)
                self.move_file(test_set,data_set_path+'//'+dir,data_set_path+'//test_set'+'//'+dir+ '//'+'subject_'+ subject)



        
    def move_file(self,files,dir,dst):
        for file in files:
            if not os.path.exists(dst):
                os.makedirs(dst)
            subject = file.split('_')[0]

            if not os.path.exists(dst ):
                os.makedirs(dst )
                print(dst )
            shutil.move(dir+'//'+file,dst)

    

if __name__ == '__main__':
    url = 'E://srt//CapgMyoData//'
    a = data_proc(url+'dbc_raw',url+'dbc_proc_python')
    #a.data_process()
    a.divide_data('E://srt//CapgMyoData//dbc_proc_python')