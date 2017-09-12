import numpy as np
import math
import xlrd
import h5py
from keras import metrics
from keras import regularizers
from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import json
from keras.utils.np_utils import to_categorical
import random

# Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

feature0=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata0.mat")
data0 = feature0['traindata0'][:]

feature1=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata1.mat")
data1 = feature1['traindata0'][:]

feature2=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata2.mat")
data2 = feature2['traindata0'][:]

feature3=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata3.mat")
data3 = feature3['traindata0'][:]

feature4=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata4.mat")
data4 = feature4['traindata0'][:]

feature5=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata5.mat")
data5 = feature5['traindata0'][:]

feature6=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata6.mat")
data6 = feature6['traindata0'][:]

feature7=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata7.mat")
data7 = feature7['traindata0'][:]

feature8=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata8.mat")
data8 = feature8['traindata0'][:]

feature9=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata9.mat")
data9 = feature9['traindata0'][:]

feature10=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata10.mat")
data10 = feature10['traindata0'][:]

feature11=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata11.mat")
data11 = feature11['traindata0'][:]

feature12=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata12.mat")
data12 = feature12['traindata0'][:]

feature13=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata13.mat")
data13 = feature13['traindata0'][:]

feature14=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata14.mat")
data14 = feature14['traindata0'][:]

feature15=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata15.mat")
data15 = feature15['traindata0'][:]

feature16=h5py.File(r"D:\大四下\换道\最新换道数据\lcdata16.mat")
data16 = feature16['traindata0'][:]

raw_obs=np.hstack((data0,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16))
raw_obs=raw_obs.T





X=np.array([raw_obs[:,0]/22, raw_obs[:,1]/2, raw_obs[:,4]/11, raw_obs[:,5]/100, raw_obs[:,6]/2, raw_obs[:,7]/4, raw_obs[:,8]/11, raw_obs[:,9]/100, raw_obs[:,10]/2, raw_obs[:,11]/4, raw_obs[:,12]/11, raw_obs[:,13]/100, raw_obs[:,14]/2, raw_obs[:,15]/4, raw_obs[:,16]/11, raw_obs[:,17]/100, raw_obs[:,18]/2, raw_obs[:,19]/4, raw_obs[:,20]/11, raw_obs[:,21]/100, raw_obs[:,22]/2, raw_obs[:,23]/4, raw_obs[:,24]/11, raw_obs[:,25]/100, raw_obs[:,26]/2, raw_obs[:,27]/4])
X=X.T

laneChange = np.zeros(len(raw_obs))
for i in range(raw_obs.shape[1]):
    if raw_obs[i,28]==1:#left change
        laneChange[i] = random.uniform(0.1739523314093953, 1-0.1739523314093953)
    if raw_obs[i,29]==1:# NO change
        laneChange[i] = random.uniform(0,0.1739523314093953)
    if raw_obs[i,30]==1:#right change
        laneChange[i] = random.uniform(1-0.1739523314093953, 1)
        
Y1=laneChange.reshape(len(raw_obs),1)

Y2=raw_obs[:,31]
a_z=np.where(Y2>0)
a_f=np.where(Y2<0)
Y2[a_z]=Y2[a_z]/3.5
Y2[a_f]=Y2[a_f]/8
Y2=Y2.reshape(len(raw_obs),1)

Y=np.hstack((Y1,Y2))
Y=np.array(Y)
#Rs = np.arange(rows)
#Rs1=np.random.permutation(Rs)
#Rtr=Rs[Rs1[:300000]]
#Rte=Rs[Rs1[300000:]]
X_train=X[:740000][:]
Y_train=Y[:740000][:]
#Y2_train=Y2[:740000]
X_test=X[740000:][:]
Y_test=Y[740000:][:]
#Y2_test=Y2[740000:]

print("Now we create neural network")
S = Input(shape=[26])
h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)

Acceleration = Dense(1, activation='tanh', use_bias=True, kernel_initializer=initializers.VarianceScaling(scale=1e-4, mode='fan_in', distribution='normal', seed=None), bias_initializer='zeros', bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(h1)
LaneChanging = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer=initializers.VarianceScaling(scale=1e-4, mode='fan_in', distribution='normal', seed=None), bias_initializer='zeros', bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(h1)
V = merge([LaneChanging,Acceleration],mode='concat')
model = Model(input=S,output=V)

model.compile(loss='mse', optimizer='sgd')
print("Now we train a network")
history = model.fit(X_train, [np.array(Y_train)], verbose=1)  # starts training
print("Now we test the network")
cost = model.evaluate(X_test, [np.array(Y_test)], verbose=1) # starts testing
with open("cost.txt", "w") as f:
    f.write(str(cost))


print("Now we save model")
model.save_weights("train_actor_lanechanging.h5", overwrite=True)
with open("train_actor_lanechanging.json", "w") as outfile:
    json.dump(model.to_json(), outfile)

