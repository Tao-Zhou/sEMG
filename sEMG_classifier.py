'''
Classifier for sEMG
'''

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import numpy as np
import scipy.io as sio
import os
import random

class sEMG_classifier(object):

    def __init__(self,fs,frame,use_pca=True):
        self.__fs = fs              #sampling frequency
        self.__frame = frame        #frame帧作为一次数据
        self.__pca = PCA()
        self.__train_data = []      #raw data
        self.__train_feature = np.array([])
        self.__train_label = np.array([])
        self.__model = svm.SVC(kernel='linear')

    def set_model(self,model_type,C=1.0,kernel='linear',k=3):
        #model_type is the name of model, these options are avalible
        #'svm': and you can set 2 params C, kernel of svm
        #'linearSVC': you can set C
        #'k_neighbor': k is the n_neighbor param
        if model_type == 'svm':
            self.__model = svm.SVC(C=C,kernel=kernel)
        elif model_type == 'linearSVC':
            self.__model = svm.LinearSVC(C=C)
        elif model_type == 'k_neighbor':
            self.__model = KNeighborsClassifier(n_neighbors=k)
        else:
            print('model_type error')
    
    def add_train_data(self,file,restart=False):
        '''
        if restart is True, this object's data will be cleared
        add mat file to train_data list
        '''
        data = sio.loadmat(file)
        if not restart:
            self.__train_data.append({'data':data['data'],'label':data['label']})
        else:
            self.__train_data = [{'data':data['data'],'label':data['label']}]

    def data_preproc(self,data):
        #zero mean process
        #data's format is [{'data': , 'label':}]
        for i in range(len(data)):
            d = data[i]['data']
            for l in range(np.shape(d)[0]):
                data[i]['data'][l] = data[i]['data'][l] - np.mean(data[i]['data'][l])
        return data


    def time_window(self,data):
        '''
        use window function to propose raw data
        window's length is frame
        if labels in one window are not the same
        this window will be abandoned
        data's format is [{'data':, 'label':} * n]
        '''
        result = {'data':[],'label':[]}
        frame = self.__frame
        for mat in data:
            label = mat['label']
            data = mat['data']
            col = np.shape(data)[1]
            num = int(np.floor(col/frame))
            for i in range(num):
                temp = data[:,i*frame:(i+1)*frame-1]
                result['data'].append(temp)
                result['label'].append(int(label))
        return result


    def feature_extract(self,windowed_data):
        #windowed_data's format:
        #{'data':[], 'label':[]}
        #each member in 'data''s list is an numpy array
        #each row is the data of one electrode 
        #return a matrix [n_examples * n_features]
        res = None
        for idx in range(len(windowed_data['data'])):
            data = windowed_data['data'][idx]
            label = windowed_data['label'][idx]
            rows = np.shape(data)[0]
            #print(rows)
            #feature extract
            MAV = np.mean(np.abs(data),axis=1)
            SSI = np.sum(data**2,axis=1)
            RMS = (np.mean(data**2,axis=1))**0.5
            #LOG = np.exp(np.mean(np.log(np.abs(data)),axis=1))
            AAC = np.mean(np.abs(np.diff(data,axis=1)),axis=1)
            MFL = np.log10((np.mean(np.abs(np.diff(data,axis=1)),axis=1))**0.5)
            MIN = np.min(data,axis=1)
            MAX = np.max(data,axis=1)
            STD = np.std(data,axis=1)
            PSD = np.abs(np.fft.rfft(data,axis=1))

            normalize = True
            if normalize:
                MAV = MAV/np.sum(MAX**2)
                SSI = SSI/np.sum(SSI**2)
                RMS = RMS/np.sum(RMS**2)
                #LOG = LOG/np.sum(LOG**2)
                AAC = AAC/np.sum(AAC**2)
                MFL = MFL/np.sum(MFL**2)
                MIN = MIN/np.sum(MIN**2)
                MAX = MAX/np.sum(MAX**2)
                STD = STD/np.sum(STD**2)
                for i in range(np.shape(PSD)[0]):
                    PSD[i,:] = PSD[i,:]/np.sum(PSD[i,:]**2)
            
            temp = np.hstack((\
                    MAV.reshape(rows,1),\
                    SSI.reshape(rows,1),\
                    RMS.reshape(rows,1),\
                    #LOG.reshape(rows,1),\
                    AAC.reshape(rows,1),\
                    MFL.reshape(rows,1),\
                    MIN.reshape(rows,1),\
                    MAX.reshape(rows,1),\
                    STD.reshape(rows,1),\
                    PSD,\
                    data)).flatten('F')
            #become a matrix 
            #print(np.shape(temp))
            if res is None: 
                res = temp
            else:
                res = np.vstack((res,temp))
            #print(np.shape(res)[0])
        #print(np.shape(res))
        return res

    
    def inner_data_proc(self):
        if len(self.__train_data) > 0:
            print('train_data process begin')
            self.__train_data = self.data_preproc(self.__train_data)
            print('pre_process done')
            windowed_data = self.time_window(self.__train_data)
            print('data windowed done')
            self.__train_feature = self.feature_extract(windowed_data)
            self.__train_label = np.array(windowed_data['label'])
            print('feature extract done')


    def fit(self):
        #process train data and make the model fit the data
        if np.shape(self.__train_feature)[0] == 0:
            self.inner_data_proc()
            idx = np.where(np.any(np.isnan(self.__train_feature),axis=1)==True)
            if(len(idx)>0):
                self.__train_feature = np.delete(self.__train_feature,idx,axis=0)
                self.__train_label = np.delete(self.__train_label,idx,axis=0)
            
        print('model training begin')
        self.__model.fit(self.__train_feature,self.__train_label)
        print('training_done!')
        #joblib.dump(self.__model,'model.m')

    def predict(self,data):
        #input is just sEMG data
        #return a list of prediction
        #each prediction corresponds to a frame of data
        data = {'data':data,'label':0}
        data = self.data_preproc(data)
        windowed_data = self.time_window(data)
        feature = self.feature_extract(windowed_data)
        return self.__model.predict(feature)

    
    def assess_on_single_file(self,file):
        # make prediction on a file
        # return the ture/false num in format (T,F)
        data = sio.loadmat(file)
        data = [data]
        data = self.data_preproc(data)
        windowed_data = self.time_window(data)
        feature = self.feature_extract(windowed_data)
        #print(feature.dtype)
        label = np.array(windowed_data['label'])
        #if NAN appears
        try:
            idxs = np.any(np.isnan(feature),axis=1)
        except TypeError:
            print('type error')
            return (0,0)
        else:
            idx = np.where(idxs==True)[0]
            if(len(idx)>0):
                feature = np.delete(feature,idx,axis=0)
                label = np.delete(label,idx,axis=0)
            prediction = self.__model.predict(feature)
            
            T = np.sum(label==prediction)
            F = len(label)-T
            return (T,F)


    def tranverse_files(self,path):
        # tranverse all files in path(not in subfolder)
        # and make predictions
        # return all files' [T,F]
        files = os.listdir(path)
        print('files len:' + str(len(files)))
        a = [0,0]
        for file in files:
            if(file[-4:]=='.mat'):
                temp = self.assess_on_single_file(os.path.join(path,file))
                a[0] = a[0] + temp[0]
                a[1] = a[1] + temp[1]
        return a

    def save_model(self,path):
        joblib.dump(self.__model,path)

    def load_model(self,file):
        self.__model = joblib.load(file)



if __name__ == '__main__':
    a = sEMG_classifier(1000,100)
    basic_url = 'E:\\srt\\CapgMyoData\\dbc_proc_python'

    subject_num = 5
    max_num = 4
    train_url = basic_url + '\\train_set'
    dirs = os.listdir(train_url)
    for dir in dirs:
        label_dir = os.path.join(train_url,dir)
        subjects_url = os.listdir(label_dir)
        for subject in subjects_url[:subject_num]:
            subject_path = os.path.join(label_dir,subject)
            files = os.listdir(subject_path)
            num = max_num
            if len(files) < max_num:
                num = len(files)
            if(dir=='0'):
                num = 10
            for file in files[:num]:
                a.add_train_data(os.path.join(subject_path,file))
                #print(file)
                    
    for model_type in ['svm','linearSVC','k_neighbor']:
        a.set_model(model_type)
        a.fit()
        a.save_model('E://python//models//'+model_type+'1.m')


        