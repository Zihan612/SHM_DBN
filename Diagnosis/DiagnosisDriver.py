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
    # Y_measurement = Output_X_new

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
sample_i = a_sample
sample_ib = b_sample
plt.plot(sample_i.T)
# plt.plot(sample_ib.T)
plt.ylim([0,4])
plt.xlabel('Steps')
plt.ylabel('Crack length')
plt.title('Sythetic crack samples')

#%%
plt.figure()
index = 612
a_true = a_sample[index,:]
b_true = b_sample[index,:]
plt.plot(a_true.T, label='True state variable')
plt.plot(b_true.T, label='Fix gap length at 50 in.')
plt.ylim([0,4])
plt.xlabel('Steps')
plt.ylabel('Crack length')
plt.title('Sythetic crack samples')
plt.legend(loc='upper left')

fig = plt.figure()
hup_true = hup_sample[index-1,:]
hdown_true = hdown_sample[index-1,:]
plt.plot(hup_true.T,'-x',hdown_true.T,'-o')
# plt.legend()
plt.title('Synthetic hydrostatic pressure measurments')

#%% Generate synthetic image measurements
noise_img = 1e-4
pixel_length = 0.5
Npic = 1 # Number of pictures taken at each time step
y_obs=[]
for io in range(Npic):
    y_temp = measurement_img(hup_true,hdown_true,l_true[:-1],a_true[:-1],pixel_length)
    y_true = y_temp + np.random.normal(0, noise_img, size=(Nt,len(y_temp[0]))) # Synthetic observations
    # y_true = y_temp # Synthetic observations
    
    if io == 0:
        y_obs = y_true
    else:
        y_obs=np.hstack((y_obs,y_true))

Nobs_img=len(y_obs[0]) # Number of observations at each time step

#%% Generate synthetic sensor measurements
noise_sensor = 1e-5

Nread = 5 # Number of sensor reading at each time step
S_obs=[]
for io in range(Nread):
    S_temp = measurement_sensor(hup_true,hdown_true,l_true[:-1])
    S_true = S_temp + np.random.normal(0, noise_sensor, size=(Nt,len(S_temp[0]))) # Synthetic observations
    if io == 0:
        S_obs = S_true
    else:
        S_obs=np.hstack((S_obs,S_true))

Nobs_sensor=len(S_obs[0]) # Number of observations at each time step
#%%
####################### Diagnostics and Prognostics ##########################
######## Right now only has diagnostics, will add prognostics later ##########
##############################################################################
# Infer state variable and unknown parameters based on 
# (1)synthetic load measurements and (2) observations of y
para_L=np.asarray([1e-4,1])
para_U=np.asarray([1e-3,3])
Nparticle=1000 # Number of particles
# Generate prior samples of unknown parameters
para_sample=np.random.uniform(para_L[0],para_U[0],size=(Nparticle,1))
# para_gap_sample = para_gap*np.ones(shape=(Nparticle,1))
for ip in range(1,len(para_L)):
    para_sample_temp=np.random.uniform(para_L[ip],para_U[ip],size=(Nparticle,1))
    # Put all prior samples together
    para_sample=np.hstack((para_sample,para_sample_temp))
    
prior_sample=copy.copy(para_sample) # Prior sample of unknown parameters
x_prior=a0*np.ones(shape=(Nparticle,1)) # Prior of state variable
l_prior=l0*np.ones(shape=(Nparticle,1)) # Prior of state variable
    
Para1_post_percentile=[]
Para2_post_percentile=[]
X_post_percentile=[]
L_post_percentile=[]

ParaGap1_post_percentile=[]
ParaGap2_post_percentile=[]
ParaGap3_post_percentile=[]

para_gap_L=np.asarray([0.49,0.038,0.59])
para_gap_U=np.asarray([0.51,0.041,0.62])
# para_gap_L=np.asarray([0.1,0.01,0.4])
# para_gap_U=np.asarray([0.9,0.09,0.7])
para_gap_sample=np.random.uniform(para_gap_L[0],para_gap_U[0],size=(Nparticle,1))
for ip in range(1,len(para_gap_L)):
    para_gap_sample_temp=np.random.uniform(para_gap_L[ip],para_gap_U[ip],size=(Nparticle,1))
    # Put all prior samples together
    para_gap_sample=np.hstack((para_gap_sample,para_gap_sample_temp))
prior_gap_sample=copy.copy(para_gap_sample) # Prior sample of unknown parameters

    
for it in range(Nt): # Time step
    print(str(it)+'/'+str(Nt))
    ds_sample=0*np.ones(shape=(Nparticle,1))
    # Propagate prior particles to state variables
    hup_temp = hup_true[it]*np.ones(shape=(Nparticle,1))
    hdown_temp = hdown_true[it]*np.ones(shape=(Nparticle,1))
    # u_temp = u_true[it]*np.ones(shape=(Nparticle,1))
    u_temp=np.random.normal(0,1,size=(Nparticle,1))
    
    l_temp=l_true[it]*np.ones(shape=(Nparticle,1))
    l_uncertainty=gap_degradation(prior_gap_sample,u_temp,l_prior,Nparticle)
    x_uncertainty=state_equation(prior_sample,hup_temp,hdown_temp,l_temp,x_prior,ds_sample,Nparticle)
    # Obtain particles of measureable variable
    # y_mean=measurement_img(hup_temp,hdown_temp,l_temp,x_uncertainty,pixel_length)
    # S_mean=measurement_sensor(hup_temp,hdown_temp,l_temp)
    y_mean=measurement_img(hup_temp,hdown_temp,l_temp,x_uncertainty,pixel_length)
    S_mean=measurement_sensor(hup_temp,hdown_temp,l_uncertainty)
    
    # Compute likelihood1 of particles
    Likelihood1=np.zeros(shape=(Nparticle,))
    for io in range(Nobs_img):
        observation_temp=y_obs[it][io]
        io_temp = int(io%((Nobs_img/Npic)))
        Likelihood1_temp=norm.pdf(observation_temp, loc=y_mean[:,io_temp], scale=noise_img)
        Likelihood1=Likelihood1+np.log(Likelihood1_temp)
        
    # Compute likelihood2 of particles
    Likelihood2=np.zeros(shape=(Nparticle,))
    for io in range(Nobs_sensor):
        observation_temp=S_obs[it][io]
        io_temp = int(io%((Nobs_sensor/Nread)))
        Likelihood2_temp=norm.pdf(observation_temp, loc=S_mean[:,io_temp], scale=noise_sensor)
        Likelihood2=Likelihood2+np.log(Likelihood2_temp)
    
    # Likelihood = 1*np.log(Likelihood1_temp)+1*np.log(Likelihood2_temp)
    Likelihood = 1*Likelihood1 + 1*Likelihood2
    # Compute weights of the particles
    Likelihood=Likelihood-np.max(Likelihood)
    if (np.max(np.exp(Likelihood))==0) and (np.min(np.exp(Likelihood))==0):
        weight_Bayes=1/len(np.exp(Likelihood))*np.ones(shape=(len(np.exp(Likelihood)),))
    else:
        weight_Bayes=np.exp(Likelihood)/sum(np.exp(Likelihood))
    ########################## Resampling #####################
    cumulative_sum = np.cumsum(weight_Bayes)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes_post = np.searchsorted(cumulative_sum, np.random.uniform(0,1,Nparticle))
    
    ###################### Obtain the posterior distribution #####################
    para_post=prior_sample[indexes_post,:]
    para_gap_post=prior_gap_sample[indexes_post,:]
    x_post=x_uncertainty[indexes_post,:]
    l_post=l_uncertainty[indexes_post,:]
    Para1_percentile=np.percentile(para_post[:,0],(5,50,95))
    Para2_percentile=np.percentile(para_post[:,1],(5,50,95))
    x_percentile=np.percentile(x_post,(5,50,95))
    l_percentile=np.percentile(l_post,(5,50,95))
    
    ParaGap1_percentile=np.percentile(para_gap_post[:,0],(5,50,95))
    ParaGap2_percentile=np.percentile(para_gap_post[:,1],(5,50,95))
    ParaGap3_percentile=np.percentile(para_gap_post[:,2],(5,50,95))
    
    
    # Store the percentile values
    Para1_post_percentile.append(Para1_percentile)
    Para2_post_percentile.append(Para2_percentile)
    X_post_percentile.append(x_percentile)
    L_post_percentile.append(l_percentile)
    
    ParaGap1_post_percentile.append(ParaGap1_percentile)
    ParaGap2_post_percentile.append(ParaGap2_percentile)
    ParaGap3_post_percentile.append(ParaGap3_percentile)
    
    ######################### Update prior distribution ##########################
    para1=np.random.normal(0,1e-6,size=(Nparticle,1))
    para2=np.random.normal(0,1e-3,size=(Nparticle,1))
    para_noise=np.hstack((para1,para2))
    prior_sample=para_post+para_noise
    
    para_gap1=np.random.normal(0,1e-4,size=(Nparticle,1))
    para_gap2=np.random.normal(0,1e-5,size=(Nparticle,1))
    para_gap3=np.random.normal(0,1e-4,size=(Nparticle,1))
    para_gap_noise=np.hstack((para_gap1,para_gap2,para_gap3))
    prior_gap_sample=para_gap_post+para_gap_noise
    
    # prior_sample=para_post
    x_prior=copy.copy(x_post)
    l_prior=copy.copy(l_post)

########################## plot posterior distribution ########################
Para1_post_percentile=np.asarray(Para1_post_percentile)
Para2_post_percentile=np.asarray(Para2_post_percentile)
X_post_percentile=np.asarray(X_post_percentile)
L_post_percentile=np.asarray(L_post_percentile)

ParaGap1_post_percentile=np.asarray(ParaGap1_post_percentile)
ParaGap2_post_percentile=np.asarray(ParaGap2_post_percentile)
ParaGap3_post_percentile=np.asarray(ParaGap3_post_percentile)

#%%
plt.figure()
plt.plot(Para1_post_percentile[:,0],'--',Para1_post_percentile[:,1],'-b',Para1_post_percentile[:,2],'--')
plt.plot([0,Nt],[para[0,0],para[0,0]],'-r')
plt.ylabel('C')
plt.title('Posterior distribution of C')
plt.legend(['5th percentile','Posterior mean','95th percentile','True'])

plt.figure()
plt.plot(Para2_post_percentile[:,0],'--',Para2_post_percentile[:,1],'-b',Para2_post_percentile[:,2],'--')
plt.plot([0,Nt],[para[0,1],para[0,1]],'-r')
plt.ylabel('m')
plt.title('Posterior distribution of m')
plt.legend(['5th percentile','Posterior mean','95th percentile','True'])

plt.figure()
plt.plot(X_post_percentile[:,0],'--',X_post_percentile[:,1],'-b',X_post_percentile[:,2],'--', a_true[1:],'-.r')
plt.ylabel('Crack length')
plt.title('Posterior distribution of crack length')
plt.legend(['5th percentile','Posterior mean','95th percentile','True'])

plt.figure()
plt.plot(L_post_percentile[:,0],'--',L_post_percentile[:,1],'-b',L_post_percentile[:,2],'--', l_true[1:],'-.r')
plt.ylabel('Gap length')
plt.title('Posterior distribution of gap length')
plt.legend(['5th percentile','Posterior mean','95th percentile','True'])


#%%
# np.save('a_predict.npy',X_post_percentile)
# np.save('a_true.npy',a_true[1:])
# np.save('para1_predict.npy',Para1_post_percentile)
# np.save('para2_predict.npy',Para2_post_percentile)
np.save('l_predict.npy',L_post_percentile)

np.save('paraGap1_predict.npy',ParaGap1_post_percentile)
np.save('paraGap2_predict.npy',ParaGap2_post_percentile)
np.save('paraGap3_predict.npy',ParaGap3_post_percentile)

# a_predict_all = np.load('a_predict.npy')
# a_predict = a_predict_all[:,1]
# para1_predict_all = np.load('para1_predict.npy')
# para1_predict = para1_predict_all[:,1]
# para2_predict_all = np.load('para2_predict.npy')
# para2_predict = para2_predict_all[:,1]
# l_predict = np.load('l_predict.npy')
#%%
# ####################### Prognostics ##########################
# ##############################################################################
# def findFS(Curve,failureLevel):
#     idx = np.argwhere(np.diff(np.sign(Curve-failureLevel)))
#     idx = idx[:,-1]
#     temp1 = np.zeros(len(idx))
#     temp2 = np.zeros(len(idx))
#     for i in range(len(idx)):
#         temp1[i] = Curve[i,idx[i]]
#         temp2[i] = Curve[i,idx[i]+1]
#     failureStep = np.divide((failureLevel-temp1),(temp2-temp1))+idx
#     return failureStep
#%%

# import numpy.matlib
# np.random.seed(123)
# Nt=100
# N = 50 # samples for prognosis
# failureLevel_crack = 3.5
# failureLevel_gap = 80
# RUL_percentile_crack=np.zeros((Nt,3))
# RUL_percentile_gap=np.zeros((Nt,3))
# # for currentStep in range(99):
# currentStep = 70
# failureStep = np.zeros((Nt,N))
# a_start = a_predict[currentStep] # the predicted crack length at current step
# para=np.asarray([[para1_predict[currentStep],para2_predict[currentStep]]]) # the predicted parameters at current step

# a_sample=a_start*np.ones(shape=(N,1))
# hup_sample=np.random.normal(550,h_range,size=(N,1))
# hdown_sample=np.random.normal(150,h_range,size=(N,1))

# remainStep = Nt-currentStep

# ds_t=0*np.ones(shape=(N,1))
# for it in range(remainStep):       
#     if it==0:
#         hup_temp=hup_sample
#         hdown_temp=hdown_sample
#     else:
#         hup_temp=np.random.normal(550,h_range,size=(N,1))
#         hdown_temp=np.random.normal(150,h_range,size=(N,1))
#         # Store data of L-->Load data
#         hup_sample=np.hstack((hup_sample,hup_temp))
#         hdown_sample=np.hstack((hdown_sample,hdown_temp))
        
#     a_temp=state_equation(para,hup_temp,hdown_temp,a_sample[:,-1],ds_t,N)
#     # Update x_sample
#     a_sample=np.hstack((a_sample,a_temp))

# a_before = np.matlib.repmat(a_predict[:currentStep],N,1)
# a_sample_full = np.append(a_before,a_sample,axis=1)
    
# failureStep_temp = findFS(a_sample_full,failureLevel)-currentStep
# # plt.hist(failureStep_temp, bins=100)
# # plt.gca().set(title='Remaining Useful Life', ylabel='Frequency')
#     # if failureStep_temp.size>0:
#     #     RUL_percentile[currentStep,:] = np.percentile(failureStep_temp,(5,50,95))
#     #     currentStep += 1


# plt.figure()
# t_forward = np.array(range(70))
# plt.plot(RUL_percentile[:70,0],'--g',RUL_percentile[:70,1],'r',RUL_percentile[:70,2],'--g', t_forward[::-1],'-k')
# plt.ylabel('Remaining Useful Life (steps)')
# plt.xlabel('steps')
# # plt.xlim(xmin=0)
# plt.ylim(ymin=0)
# plt.title('Remaining Useful Life (RUL) for crack')
# plt.legend(['Upper confidence limit (95%)','Mean prediction','Lower confidence limit (5%)','True RUL'])
