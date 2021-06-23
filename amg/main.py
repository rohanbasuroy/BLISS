#multiple BOs, kripke
import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer,Categorical
from skopt.utils import use_named_args
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct,ConstantKernel,WhiteKernel)
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
import sklearn.gaussian_process as gp
import subprocess as sp
import shlex
import random
import numpy as np
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter
import statistics
import time 
import os 
FNULL = open(os.devnull, 'w')

#define parameters
max_calls=200
num_initial_sample=1
percentage_sampled_by_acq=0.5
delay_min=20 #(delay_min - delay_window) must be greater than the (number of models + num_initial_smaple)
delay_max=50
delay_window=4
lookahead_max=10
lookahead_window=4
throttling_times=2.5


param_list=[]
exe_list=[]
n_list=["cljp","cgc","pmis","hmis","ruge"]
n_choice=[0,1,2,3,4]
s_choice=[0,1,2,3,4,5,6,7,8,9,10,13,14,15,18,50,51,60,61]
r_choice=[0,1,3,4,13,16,20]
i_choice=[0,1,2,3,4,6,8,12]
p_choice=[2,4,6]
u_choice=[0.8,1.0,1.2,1.5,1.7,2.0,2.3,2.5,2.8,3.0]
#u_choice=[3.0]
f_choice=[0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2]
#f_choice=[2.2]
h_choice=[0,1]
#h_choice=[1]
param_choice=[n_choice, s_choice, r_choice, i_choice, p_choice, u_choice, f_choice, h_choice]


def hyperthreading(i):
    if i==0: #off
        j=10
        while j <20:
            comm="sudo bash -c 'echo 0 > /sys/devices/system/cpu/cpu"+str(j)+"/online'"
            os.system(comm)
            j+=1
    if i==1: #on
        j=10
        while j<20:
            comm="sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpu"+str(j)+"/online'"
            os.system(comm)
            j+=1
th_list=[]
def objective(list,delay):
    n=n_list[int(list[0])]
    s=int(list[1])
    r=int(list[2])
    i=int(list[3])
    p=int(list[4])
    u=list[5]
    f=list[6]
    h=int(list[7])
    hyperthreading(h)
    u_comm="likwid-setFrequencies -umax "+str(u)
    sp.check_output(shlex.split(u_comm), stderr=FNULL)
    f_comm="likwid-setFrequencies -c S0 -f "+str(f)
    sp.check_output(shlex.split(f_comm), stderr=FNULL)
    if len(exe_list)>delay: #throttling only after maturity
        command="timeout "+str(throttling_times*min(exe_list))+" "+"./ij -laplacian -27pt -n 50 50 50 "+"-solver"+" "+str(s)+" "+"-rlx"+" "+str(r)+" "+"-interptype"+" "+str(i)+" "+"-"+n+" "+"-Pmx"+" "+str(p)
	th_list.append(1)
    
    else:
        command="./ij -laplacian -27pt -n 50 50 50 "+"-solver"+" "+str(s)+" "+"-rlx"+" "+str(r)+" "+"-interptype"+" "+str(i)+" "+"-"+n+" "+"-Pmx"+" "+str(p)
    #command="OMP_NUM_THREADS="+str(t)+" "+"./kripke.exe --groups 64 --quad 192 --layout "+n+" "+"--gset "+str(g)+" "+"--dset "+str(d)
    start_time=time.time()
    os.system(command)
    end_time=time.time()
    exe_time=end_time-start_time
    print ("the execution time is", exe_time)
    if exe_time < 0.6:
	return -1.0*10
    else:
    	param_list.append([n,s,r,i,p,u,f,h])
   	exe_list.append(exe_time)
    	return -1.0*exe_time
    
    #print (n,g,d,t,u)
    #r=(random.randint(1,10))
    #exe_list.append(r)
    #return (r)

def surrogate(model, XX):
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(XX, return_std=True)
def acquisition_ei(XX, Xsamples, model):
    yhat, _ = surrogate(model, XX)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    mu = mu[:]
    Z=((mu - best) / (std+1E-9))
    #print (mu - best)* norm.cdf(Z) + std*norm.pdf(Z)
    return (mu - best)* norm.cdf(Z) + std*norm.pdf(Z)
def acquisition_pi(XX, Xsamples, model):
    yhat, _ = surrogate(model, XX)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    mu = mu[:]
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs
def acquisition_ucb(XX, Xsamples, model):
    yhat, _ = surrogate(model, XX)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    mu = mu[:]
    v=1
    delta=0.1
    d=len(param_choice)
    t=len(exe_list)
    Kappa = np.sqrt( v* (2*  np.log( (t**(d/2. + 2))*(np.pi**2)/(3. * delta)  )))
    return mu + Kappa*(std+1E-9)
    
def opt_acquisition(XX, yy, model, acqval):
    total_choice=1
    for cc in range(len(param_choice)):
        total_choice*=len(param_choice[cc])
    Xsamples=[]
    out_count=0
    while out_count < percentage_sampled_by_acq*total_choice:
        in_list=[]
        in_count=0
        while in_count < len(param_choice):
            in_list.append(random.choice(param_choice[in_count]))
            in_count+=1
        if in_list not in XX :
            Xsamples.append(in_list)
        out_count+=1
    if acqval==0:
        scores = acquisition_ei(XX, Xsamples, model)
    if acqval==1:
        scores = acquisition_pi(XX, Xsamples, model)
    if acqval==2:
        scores = acquisition_ucb(XX, Xsamples, model)
    ix = argmax(scores)
    return Xsamples[ix] 

def get_lookahead_status(model,delay):
    
    if len(exe_list)<delay:
        return 0
    else:
        lookahead_selection_list=[]
        j=0
        while j < lookahead_window:
            xx = opt_acquisition(XX, yy, model, acqval)
            actual = objective(xx,delay)
            est, _ = surrogate(model, [xx])
            XX.append(xx)
            yy.append(actual)
            model.fit(XX, yy)
	    print (est[0], actual)  ##
            if abs(abs(est[0])-abs(actual))< abs(actual):
                lookahead_selection_list.append(int((1-(abs(abs(est[0])-abs(actual))/abs(actual)))*lookahead_max))
            else:
                lookahead_selection_list.append(0)
            j+=1
        lookahead=int(statistics.mean(lookahead_selection_list))
        lookahead_list.append(lookahead)
        return lookahead
        


kernel_options=[gp.kernels.DotProduct() + gp.kernels.WhiteKernel(), gp.kernels.Matern(length_scale=1.0, nu=1.5),
                gp.kernels.RBF(length_scale=1.0),gp.kernels.RationalQuadratic(length_scale=1.0),
                gp.kernels.ExpSineSquared(length_scale=1.0)]
#gp.kernels.ExpSineSquared(length_scale=1.0)

#model1-4:EI, model5-8:PI, model9-12:UCB
model1 = GaussianProcessRegressor(kernel=kernel_options[0])
model2 = GaussianProcessRegressor(kernel=kernel_options[1])
model3 = GaussianProcessRegressor(kernel=kernel_options[2])
model4 = GaussianProcessRegressor(kernel=kernel_options[3])
model5 = GaussianProcessRegressor(kernel=kernel_options[0])
model6 = GaussianProcessRegressor(kernel=kernel_options[1])
model7 = GaussianProcessRegressor(kernel=kernel_options[2])
model8 = GaussianProcessRegressor(kernel=kernel_options[3])
model9 = GaussianProcessRegressor(kernel=kernel_options[0])
model10 = GaussianProcessRegressor(kernel=kernel_options[1])
model11 = GaussianProcessRegressor(kernel=kernel_options[2])
model12 = GaussianProcessRegressor(kernel=kernel_options[3])

model_list=[model1,model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12]
#model_list=[model1, model5, model9]
model_sampling_list=[[] for i in range(0,len(model_list))]


##reset hardware knobs
hyperthreading(1)
u_comm="likwid-setFrequencies -umax 3.0"
sp.check_output(shlex.split(u_comm), stderr=FNULL)
f_comm="likwid-setFrequencies -c S0 -f 2.2"
sp.check_output(shlex.split(f_comm), stderr=FNULL)

delay=9999 #initially set it to an arbitary large value
lookahead_counter=0
#initial sampling 
XX=[]
out_count=0
while out_count < num_initial_sample:
    in_list=[]
    in_count=0
    while in_count < len(param_choice):
        in_list.append(random.choice(param_choice[in_count]))
        in_count+=1
    XX.append(in_list)
    out_count+=1
yy = [objective(xx,delay) for xx in XX]
for m in model_list:
    m.fit(XX, yy)
    

mm=[]
model_selection_list=[]
delay_selection_list=[]
delay_list=[]
lookahead_list=[]

i=0
while len(exe_list)<max_calls:
    model_min_list=[]
    if i==0:
        for model in model_list:
            if model == model1 or model == model2 or model == model3 or model == model4:
                acqval=0
            elif model == model5 or model == model6 or model == model7 or model == model8:
                acqval=1
            elif model == model9 or model == model10 or model == model11 or model == model12:
                acqval=2
            xx = opt_acquisition(XX, yy, model, acqval)
            actual = objective(xx,delay)
            model_sampling_list[model_list.index(model)].append(-1*actual)
            est, _ = surrogate(model, [xx])
            XX.append(xx)
            yy.append(actual)
            model.fit(XX, yy)
            if model==model1:
                mm.append("m1")
            elif model==model2:
                mm.append("m2")
            elif model==model3:
                mm.append("m3")
            elif model==model4:
                mm.append("m4")
            elif model==model5:
                mm.append("m5")
            elif model==model6:
                mm.append("m6")
            elif model==model7:
                mm.append("m7")
            elif model==model8:
                mm.append("m8")
            elif model==model9:
                mm.append("m9")
            elif model==model10:
                mm.append("m10")
            elif model==model11:
                mm.append("m11")
            elif model==model12:
                mm.append("m12")
            model_selection_list.append(model)
            i+=1
    else:
        
        model=random.choice(model_selection_list)                   
        if lookahead_counter != 0:  #prediction, no incrementation, no appending to model_selection_list 
            if model == model1 or model == model2 or model == model3 or model == model4:
                acqval=0
            elif model == model5 or model == model6 or model == model7 or model == model8:
                acqval=1
            elif model == model9 or model == model10 or model == model11 or model == model12:
                acqval=2
            xx = opt_acquisition(XX, yy, model, acqval)
            est, _ = surrogate(model, [xx])
            XX.append(xx)
            yy.append(est[0])
            model.fit(XX, yy) 
            lookahead_counter-=1
            
        elif lookahead_counter==0:
            if model == model1 or model == model2 or model == model3 or model == model4:
                acqval=0
            elif model == model5 or model == model6 or model == model7 or model == model8:
                acqval=1
            elif model == model9 or model == model10 or model == model11 or model == model12:
                acqval=2
            xx = opt_acquisition(XX, yy, model, acqval)
            actual = objective(xx,delay)
            model_sampling_list[model_list.index(model)].append(-1*actual)
            est, _ = surrogate(model, [xx])
            XX.append(xx)
            yy.append(actual)
            model.fit(XX, yy)
            for m in model_sampling_list:
                model_min_list.append(min(m))
            model_selection_list.append(model_list[model_min_list.index(min(model_min_list))])
            if model==model1:
                mm.append("m1")
            elif model==model2:
                mm.append("m2")
            elif model==model3:
                mm.append("m3")
            elif model==model4:
                mm.append("m4")
            elif model==model5:
                mm.append("m5")
            elif model==model6:
                mm.append("m6")
            elif model==model7:
                mm.append("m7")
            elif model==model8:
                mm.append("m8")
            elif model==model9:
                mm.append("m9")
            elif model==model10:
                mm.append("m10")
            elif model==model11:
                mm.append("m11")
            elif model==model12:
                mm.append("m12")
            i+=1
            lookahead_counter=get_lookahead_status(model,delay)
        
            
    #getting maturity/delay        
    if len(exe_list) >= delay_min-delay_window and len(exe_list) < delay_min :
        if abs(abs(est[0])-abs(actual))< abs(actual):
            delay_selection_list.append(int((abs(abs(est[0])-abs(actual))/abs(actual))*(delay_max-delay_min)))
        else:
            delay_selection_list.append(delay_max)
    if len(exe_list)==delay_min:
        delay=delay_min+int(statistics.mean(delay_selection_list))
        delay_list.append(delay)
        print ("The delay is ", delay)
        
    with open('param_list.txt', 'w') as f:
   	 for item in param_list:
        	print >> f, item
    with open('exe_list.txt', 'w') as f:
   	 for item in exe_list:
        	print >> f, item
    with open('model_list.txt', 'w') as f:
   	 for item in mm:
        	print >> f, item
    with open('delay_list.txt', 'w') as f:
   	 for item in delay_list:
        	print >> f, item
    with open('lookahead_list.txt', 'w') as f:
    	for item in lookahead_list:
        	print >> f, item


##reset hardware knobs
hyperthreading(1)
u_comm="likwid-setFrequencies -umax 3.0"
sp.check_output(shlex.split(u_comm), stderr=FNULL)
f_comm="likwid-setFrequencies -c S0 -f 2.2"
sp.check_output(shlex.split(f_comm), stderr=FNULL)

print (th_list)
