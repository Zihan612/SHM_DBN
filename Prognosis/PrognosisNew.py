# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 01:32:13 2021

@author: z5wu
"""
import sys
import os

import numpy as np

import pickle
from PIL import Image
import matplotlib.pyplot as plt

from scipy.stats import norm
import copy

np.random.seed(612)
#%% Set the scripPath and workPath

scriptPath = os.path.dirname(os.path.abspath(__file__)) + "\\"
sys.path.append(scriptPath)
os.chdir(scriptPath)

XPath= scriptPath + "Model_U1\\" # Directory for Img Surrogate in X direction
YPath= scriptPath + "Model_U3\\" # Directory for Img Surrogate in Y direction
SensorPath = scriptPath + "SensorModel\\" # Directory for Sensor Surrogate


#%% Define SIF function

ModelSIF = pickle.load(open('GlobalModel.sav', 'rb'))

def getSIF(h_up,h_down,l,a):
    
    h_up=np.reshape(h_up,(-1,1))
    h_down=np.reshape(h_down,(-1,1))
    l=np.reshape(l,(-1,1))
    a=np.reshape(a,(-1,1))
    
    if h_up.size>0 and h_down.size>0:
        SIF_array = ModelSIF.predict(np.concatenate((h_up,h_down,l,a),axis=1))
    else:
        SIF_array = []
        
    return SIF_array


#%% Define state equations
def state_equation(para,h_up,h_down,l,a0,ds,Ns):
    # para: crack growth model parameters

    # a0 is the inital state
    # Ns is the number of samples
    SIF=np.zeros(shape=(Ns,1))
    h_up=np.reshape(h_up,(-1,1))
    h_down=np.reshape(h_down,(-1,1))
    l=np.reshape(l,(-1,1))
    a0=np.reshape(a0,(-1,1))
    index0=np.where(ds==0) # Damage stage 0
    
    SIF[index0[0],0]=getSIF(h_up[index0[0],0],h_down[index0[0],0],l[index0[0],0],a0[index0[0],0])

    C=np.reshape(para[:,0],(len(para[:,0]),1))
    m=np.reshape(para[:,1],(len(para[:,1]),1))
    a1=a0+C*SIF**m
    
    return a1

#%% Define gap degradation model
def gap_degradation(para,u,l0,Ns):
    # para: gap degradation model parameters
    
    # l0 is the inital state
    # Ns is the number of samples
    u=np.reshape(u,(-1,1))
    l0=np.reshape(l0,(-1,1))
    
    Sig=np.reshape(para[:,0],(len(para[:,0]),1))
    Q=np.reshape(para[:,1],(len(para[:,1]),1))
    mu=np.reshape(para[:,2],(len(para[:,2]),1))
    l1=l0+np.exp(Sig*u)*Q*np.power(l0,mu)
    
    return l1


# Assume true value of para
para_gap=np.asarray([[0.5,0.04,0.6]])
Nt=100 # Number of time steps for simualtion
N=1000 # Number of MCS samples
l0=50 # Initial gap length

##############################################################################
########################### Generate synthetic data ##########################
##############################################################################
l_sample=l0*np.ones(shape=(N,1))

u_sample=np.random.normal(0,1,size=(N,1))


for it in range(Nt):
    if it==0:
       u_temp=u_sample 
    else:
       u_temp=np.random.normal(0,1,size=(N,1))
       # Store data of L-->Load data
       u_sample=np.hstack((u_sample,u_temp))
    l_temp=gap_degradation(para_gap,u_temp,l_sample[:,-1],N)
    # Update x_sample
    l_sample=np.hstack((l_sample,l_temp))


plt.figure()
plt.plot(l_sample.T)
plt.xlabel('Steps')
plt.ylabel('Gap length (inch)')
plt.title('All realizations of state variable')
# plt.ylim([0,4])
#%%
index_gap = 2
u_true = u_sample[index_gap,:]   
l_true = l_sample[index_gap]
plt.plot(l_true)
plt.xlabel('Steps')
plt.ylabel('Gap length (inch)')
plt.title('Gap length variable')

#%% Define image measurement equations
# load the model from disk
os.chdir(XPath)
ModelX = pickle.load(open('Camera_Model_U1.pkl', 'rb'))
ModelX1 = ModelX[2]
ModelX2 = ModelX[3]
ModelX3 = ModelX[4]
ModelX4 = ModelX[5]
decoderX = ModelX[1]

os.chdir(YPath)
ModelY = pickle.load(open('Camera_Model_U3.pkl', 'rb'))
ModelY1 = ModelY[2]
ModelY2 = ModelY[3]
ModelY3 = ModelY[4]
ModelY4 = ModelY[5]
decoderY = ModelY[1]

os.chdir(scriptPath)
def measurement_img(h_up,h_down,l,a,pixel_length):
    
    h_up=np.reshape(h_up,(-1,1))
    h_down=np.reshape(h_down,(-1,1))
    l=np.reshape(l,(-1,1))
    a=np.reshape(a,(-1,1))
    N=len(h_up) # Number of MCS samples
    nSidePixel1 = int(10/pixel_length)
    nSidePixel3 = int(12.667/pixel_length)
    InputSpace = np.concatenate((h_up,h_down,l,a),axis=1)

    Reduced_OutputX1 = ModelX1.predict(InputSpace,return_std=False).reshape(-1,1)
    Reduced_OutputX2 = ModelX2.predict(InputSpace,return_std=False).reshape(-1,1)
    Reduced_OutputX3 = ModelX3.predict(InputSpace,return_std=False).reshape(-1,1)
    Reduced_OutputX4 = ModelX4.predict(InputSpace,return_std=False).reshape(-1,1)

    
    Output_X = np.concatenate((Reduced_OutputX1,Reduced_OutputX2,Reduced_OutputX3,Reduced_OutputX4),axis=1)
    Output_X_decoded = Output_X @ decoderX
    Output_X_new = np.zeros((N,nSidePixel1*nSidePixel3))
    for Xi in range(N):
        IMG_temp = np.reshape(Output_X_decoded[Xi,:],(200,253))
        IMG_resize = np.array(Image.fromarray(IMG_temp).resize((nSidePixel1,nSidePixel3)))
        IMG_array = np.reshape(IMG_resize,(1,-1))
        Output_X_new[Xi,:] = IMG_array
    
    Reduced_OutputY1 = ModelY1.predict(InputSpace,return_std=False).reshape(-1,1)
    Reduced_OutputY2 = ModelY2.predict(InputSpace,return_std=False).reshape(-1,1)
    Reduced_OutputY3 = ModelY3.predict(InputSpace,return_std=False).reshape(-1,1)
    Reduced_OutputY4 = ModelY4.predict(InputSpace,return_std=False).reshape(-1,1)

    
    Output_Y = np.concatenate((Reduced_OutputY1,Reduced_OutputY2,Reduced_OutputY3,Reduced_OutputY4),axis=1)
    Output_Y_decoded = Output_Y @ decoderY
    Output_Y_new = np.zeros((N,nSidePixel1*nSidePixel3))
    for Yi in range(N):
        IMG_temp = np.reshape(Output_Y_decoded[Yi,:],(200,253))
        IMG_resize = np.array(Image.fromarray(IMG_temp).resize((nSidePixel1,nSidePixel3)))
        IMG_array = np.reshape(IMG_resize,(1,-1))
        Output_Y_new[Yi,:] = IMG_array
    
    Y_measurement = np.append(Output_X_new,Output_Y_new,axis=1)
    # Y_measurement = Output_Y_new

    return Y_measurement

#%% Define sensor measurement equations
os.chdir(SensorPath)
Model_Sensor = pickle.load(open('Sensor_Model.pkl', 'rb'))
ModelS1 = Model_Sensor[0]
ModelS2 = Model_Sensor[1]
ModelS3 = Model_Sensor[2]
ModelS4 = Model_Sensor[3]
# ModelS5 = Model_Sensor[4]

os.chdir(scriptPath)
def measurement_sensor(h_up,h_down,l):
    h_up=np.reshape(h_up,(-1,1))
    h_down=np.reshape(h_down,(-1,1))
    l=np.reshape(l,(-1,1))
    InputSpace = np.concatenate((l,h_up,h_down),axis=1)

    OutputS1 = ModelS1.predict(InputSpace,return_std=False).reshape(-1,1)
    OutputS2 = ModelS2.predict(InputSpace,return_std=False).reshape(-1,1)
    OutputS3 = ModelS3.predict(InputSpace,return_std=False).reshape(-1,1)
    OutputS4 = ModelS4.predict(InputSpace,return_std=False).reshape(-1,1)
    # OutputS5 = ModelS5.predict(InputSpace,return_std=False).reshape(-1,1)

    
    Output_S = np.concatenate((OutputS1,OutputS2,OutputS3,OutputS4),axis=1)
    
    return Output_S

#%%
# Assume true value of para
para=np.asarray([[3e-4,2.2]])
Nt=100 # Number of times steps for simualtion
N=1000 # Number of MCS samples
a0=1 # Initial crack length

##############################################################################
########################### Generate synthetic data ##########################
##############################################################################
a_sample=a0*np.ones(shape=(N,1))
b_sample=a0*np.ones(shape=(N,1))
h_range = 15
# hup_sample=np.random.normal(576,h_range,size=(N,1))
# hdown_sample=np.random.normal(240,h_range,size=(N,1))
hup_sample=np.random.normal(550,h_range,size=(N,1))
hdown_sample=np.random.normal(150,h_range,size=(N,1))



ds_t=0*np.ones(shape=(N,1))
for it in range(Nt): 
        
    if it==0:
       hup_temp=hup_sample
       hdown_temp=hdown_sample 
    else:
       hup_temp=np.random.normal(550,h_range,size=(N,1))
       hdown_temp=np.random.normal(150,h_range,size=(N,1))
       # Store data of L-->Load data
       hup_sample=np.hstack((hup_sample,hup_temp))
       hdown_sample=np.hstack((hdown_sample,hdown_temp))
    
    l =  l_true[it]*np.ones(shape=(N,1))
    a_temp=state_equation(para,hup_temp,hdown_temp,l,a_sample[:,-1],ds_t,N)
    # Update x_sample
    a_sample=np.hstack((a_sample,a_temp))
    
    b_temp=state_equation(para,hup_temp,hdown_temp,50*np.ones(shape=(N,1)),b_sample[:,-1],ds_t,N)
    b_sample=np.hstack((b_sample,b_temp))
#%%
plt.figure()
index = 612
a_true = a_sample[index,:]
b_true = b_sample[index,:]
plt.plot(a_true.T)
# plt.plot(b_true.T, label='Fix gap length at 50 in.')
plt.ylim([0,4])
plt.xlabel('Steps')
plt.ylabel('Crack length')
plt.title('Sythetic crack samples')
plt.legend(loc='upper left')

# fig = plt.figure()
hup_true = hup_sample[index-1,:]
hdown_true = hdown_sample[index-1,:]
# plt.plot(hup_true.T,'-x',hdown_true.T,'-o')
# plt.legend()
# plt.title('Synthetic hydrostatic pressure measurments')

#%%
# # ####################### Prognostics ##########################
# # ##############################################################################
import numpy as np
a_predict_all = np.load('a_predict.npy')
a_predict = a_predict_all[:,1]
para1_predict_all = np.load('para1_predict.npy')
para1_predict = para1_predict_all[:,1]
para2_predict_all = np.load('para2_predict.npy')
para2_predict = para2_predict_all[:,1]
l_predict_all = np.load('l_predict.npy')
l_predict = l_predict_all[:,1]


paraGap1_predict_all = np.load('paraGap1_predict.npy')
paraGap1_predict = paraGap1_predict_all[:,1]
paraGap2_predict_all = np.load('paraGap2_predict.npy')
paraGap2_predict = paraGap2_predict_all[:,1]
paraGap3_predict_all = np.load('paraGap3_predict.npy')
paraGap3_predict = paraGap3_predict_all[:,1]


def findFS(Curve,failureLevel):
    idx = np.argwhere(np.diff(np.sign(Curve-failureLevel)))
    idx = idx[:,-1]
    temp1 = np.zeros(len(idx))
    temp2 = np.zeros(len(idx))
    for i in range(len(idx)):
        temp1[i] = Curve[i,idx[i]]
        temp2[i] = Curve[i,idx[i]+1]
    failureStep = np.divide((failureLevel-temp1),(temp2-temp1))+idx
    return failureStep
#%%

import numpy.matlib
np.random.seed(555)
Nt=200
N = 2000 # samples for prognosis
failureLevel_crack = 3
failureLevel_gap = 100
# crack_post_percentile=[]
# gap_post_percentile=[]
joint_post_percentile=[]

# Optimization
# crack_Pr=[]
# gap_Pr=[]
joint_Pr=[]


for currentStep in range(88):

    # currentStep = 80
    # failureStep_crack = np.zeros((Nt,N))
    # failureStep_gap = np.zeros((Nt,N))
    
    l_start = l_predict[currentStep] # the predicted gap length at current step
    l_sample=l_start*np.ones(shape=(N,1))
    # para_gap = np.asarray([[0.5,0.04,0.6]])*np.ones(shape=(N,1))
    para_gap = np.asarray([[paraGap1_predict[currentStep],paraGap2_predict[currentStep],paraGap3_predict[currentStep]]])
    u_sample=np.random.normal(0,1,size=(N,1))
    
    
    a_start = a_predict[currentStep] # the predicted crack length at current step
    para=np.asarray([[para1_predict[currentStep],para2_predict[currentStep]]]) # the predicted parameters at current step
    
    a_sample=a_start*np.ones(shape=(N,1))
    hup_sample=np.random.normal(550,h_range,size=(N,1))
    hdown_sample=np.random.normal(150,h_range,size=(N,1))
    
    remainStep = Nt-currentStep
    
    ds_t=0*np.ones(shape=(N,1))
    for it in range(Nt):       
        if it==0:
            hup_temp=hup_sample
            hdown_temp=hdown_sample
            u_temp=u_sample
        else:
            hup_temp=np.random.normal(550,h_range,size=(N,1))
            hdown_temp=np.random.normal(150,h_range,size=(N,1))
            u_temp=np.random.normal(0,1,size=(N,1))
            # Store data of L-->Load data
            hup_sample=np.hstack((hup_sample,hup_temp))
            hdown_sample=np.hstack((hdown_sample,hdown_temp))
            u_sample=np.hstack((u_sample,u_temp))
         
        l_temp = gap_degradation(para_gap,u_temp,l_sample[:,-1],N)
        a_temp = state_equation(para,hup_temp,hdown_temp,l_temp,a_sample[:,-1],ds_t,N)
        # Update l_sample
        l_sample=np.hstack((l_sample,l_temp))
        # Update x_sample
        a_sample=np.hstack((a_sample,a_temp))
    
    l_before = np.matlib.repmat(l_predict[:currentStep],N,1)
    l_sample_full = np.append(l_before,l_sample,axis=1)
    a_before = np.matlib.repmat(a_predict[:currentStep],N,1)
    a_sample_full = np.append(a_before,a_sample,axis=1)
    
    failureStep_gap_temp = findFS(l_sample_full,failureLevel_gap)-currentStep
    failureStep_crack_temp = findFS(a_sample_full,failureLevel_crack)-currentStep
    failureStep_joint_temp = np.zeros((N,))
    for i in range(N):
        if failureStep_gap_temp[i,]<failureStep_crack_temp[i,]:
                failureStep_joint_temp[i,] = failureStep_gap_temp[i,]
        else:
            failureStep_joint_temp[i,] = failureStep_crack_temp[i,]
        
        # plt.hist(failureStep_joint_temp, bins=100)
        # plt.gca().set(title='Remaining Useful Life', ylabel='Frequency')
    
    # crack_percentile=np.percentile(failureStep_crack_temp,(5,50,95))
    # gap_percentile=np.percentile(failureStep_gap_temp,(5,50,95))
    joint_percentile=np.percentile(failureStep_joint_temp,(5,50,95))


    # Store the percentile values
    # crack_post_percentile.append(crack_percentile)
    # gap_post_percentile.append(gap_percentile)
    joint_post_percentile.append(joint_percentile)
    
    # crack_Pr.append(failureStep_crack_temp)
    # gap_Pr.append(failureStep_gap_temp)
    joint_Pr.append(failureStep_joint_temp)

    currentStep += 1


#%%
# crack_post_percentile = np.asarray(crack_post_percentile)
# gap_post_percentile = np.asarray(gap_post_percentile)
joint_post_percentile = np.asarray(joint_post_percentile)

# crack_Pr = np.asarray(crack_Pr)
# gap_Pr = np.asarray(gap_Pr)
joint_Pr = np.asarray(joint_Pr)

# np.save('crack_Pr.npy',crack_Pr)
# np.save('gap_Pr.npy',gap_Pr)
np.save('joint_Pr.npy',joint_Pr)
np.save('joint_post.npy',joint_post_percentile)


#%%
# np.save('joint_post.npy',joint_post_percentile)
#%%
# plt.figure()
t_forward = np.array(range(83))
# plt.subplot(2,1,1)
# # plt.plot(RUL_percentile_gap[:70,0],'--g',RUL_percentile_gap[:70,1],'r',RUL_percentile_gap[:70,2],'--g', t_forward[::-1],'-k')
# plt.ylabel('Remaining Useful Life (steps)')
# plt.xlabel('steps')
# # plt.xlim(xmin=0)
# plt.ylim(ymin=0)
# plt.title('Remaining Useful Life (RUL) for gap')
# plt.legend(['Upper confidence limit (95%)','Mean prediction','Lower confidence limit (5%)','True RUL'])
# plt.subplot(2,1,2)
joint_post_percentile=np.load('joint_post.npy')
plt.plot(joint_post_percentile[:,0],'--g',joint_post_percentile[:,1],'r',joint_post_percentile[:,2],'--g', t_forward[::-1],'-k')
plt.ylabel('Remaining Useful Life (steps)')
plt.xlabel('steps')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.title('Joint Remaining Useful Life (RUL)')
plt.legend(['Upper confidence limit (95%)','Mean prediction','Lower confidence limit (5%)','True RUL'])
