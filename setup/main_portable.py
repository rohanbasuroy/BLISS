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
import math
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
max_calls=75
num_initial_sample=1
percentage_sampled_by_acq=0.5
delay_min=20 #(delay_min - delay_window) must be greater than the (number of models + num_initial_smaple)
delay_max=50
delay_window=4
lookahead_max=10
lookahead_window=4
throttling_times=2.5
number_of_times_of_min_of_previous_architecture=2
diameter_of_pruning_wrt_max_sw_dist =  0.05 #5%
max_num_pruning_try=10

param_list=[]
exe_list=[]
n_list=["DGZ","DZG","GDZ","GZD","ZDG","ZGD"]
n_choice=[0,1,2,3,4,5]
g_choice=[1,2,4,8,16,32]
d_choice=[8,16,24,32,48,64,96]
t_choice=[1,2,3,4,6,8,10,12,14,16,18,20]
u_choice=[0.8,1.0,1.2,1.5,1.7,2.0,2.3,2.5,2.8,3.0]
#u_choice=[3.0]
f_choice=[0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2]
#f_choice=[2.2]
h_choice=[0,1]
#h_choice=[1]
param_choice=[n_choice, g_choice, d_choice, t_choice, u_choice, f_choice, h_choice]
pruned_config_list=[]


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


#determine which parameter configurations to not sample
def check_if_pruned(list):
    n_index_from_list=n_choice.index(list[0])
    g_index_from_list=g_choice.index(list[1])
    d_index_from_list=d_choice.index(list[2])
    t_index_from_list=t_choice.index(list[3])
    max_sw_dist = len(n_choice) + len(g_choice) + len(d_choice) + len(t_choice)
    checking_list=[n_index_from_list, g_index_from_list,d_index_from_list,t_index_from_list]
    
    with open (r"./exe_list.txt", 'r') as f:
        exe_lines=f.read().splitlines()
    for i in range (0,len(exe_lines)):
        exe_lines[i]=float(exe_lines[i])
    
    with open (r"./param_list.txt", 'r') as f:
        lines=f.read().splitlines()

    previous_sampled=[]
    count = 0
    while count < len(lines):
         inner_line = lines[count][1:-1].split(",")
         inner_list=[]
         i =0
         while i < len (inner_line)-3:     #only software params n,g,d,t
            if i == 0:
                inner_list.append(param_choice[i][n_list.index(inner_line[0][1:-1])])
            else:
                inner_list.append(param_choice[i].index(int(inner_line[i])))

            i+=1
         if exe_lines[count] < number_of_times_of_min_of_previous_architecture * min(exe_lines):
             previous_sampled.append(inner_list)
         count+=1
    #print (previous_sampled)
    ret_val=1 #not_pruned
    count = 0
    while count < len(previous_sampled):
        d=0
        in_count=0
        while in_count<len(previous_sampled[count]):
            d+=(previous_sampled[count][in_count]-checking_list[in_count])**2
            in_count+=1
        if math.sqrt(d) < diameter_of_pruning_wrt_max_sw_dist*max_sw_dist:
            ret_val=0 #to_be_pruned
            break
        count +=1
    return ret_val

#BLISS objective function
def objective(list,delay):
    n=n_list[int(list[0])]
    g=int(list[1])
    d=int(list[2])
    t=int(list[3])
    u=list[4]
    f=list[5]
    h=int(list[6])
    hyperthreading(h)
    u_comm="likwid-setFrequencies -umax "+str(u)
    sp.check_output(shlex.split(u_comm), stderr=FNULL)
    f_comm="likwid-setFrequencies -c S0 -f "+str(f)
    sp.check_output(shlex.split(f_comm), stderr=FNULL)
    os.environ["OMP_NUM_THREADS"]=str(t)
    if len(exe_list)>delay: #throttling only after maturity
        command="timeout "+str(throttling_times*min(exe_list))+" "+"./kripke.exe --groups 64 --quad 192 --layout "+n+" "+"--gset "+str(g)+" "+"--dset "+str(d)
	th_list.append(1)
    else:
        command="./kripke.exe --groups 64 --quad 192 --layout "+n+" "+"--gset "+str(g)+" "+"--dset "+str(d)    
    #command="OMP_NUM_THREADS="+str(t)+" "+"./kripke.exe --groups 64 --quad 192 --layout "+n+" "+"--gset "+str(g)+" "+"--dset "+str(d)
    start_time=time.time()
    os.system(command)
    end_time=time.time()
    exe_time=end_time-start_time
    print ("the execution time is", exe_time)
    param_list.append([n,g,d,t,u,f,h])
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
    
def opt_acquisition(XX, yy, model, acqval, true_eval):
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
        
    scores_copy=scores   
        
    if true_eval == 0:                                 #when prediction
        ix = argmax(scores_copy)
        return Xsamples[ix] 
    else:
        num_of_rounds = 0                              #when evaluation
        while num_of_rounds < max_num_pruning_try:
            ix = argmax(scores_copy)
            res=check_if_pruned(Xsamples[ix])
            if res == 1:
                return Xsamples[ix]
                break
            else:
                pruned_config_list.append(Xsamples[ix])
                scores_copy[argmax(scores_copy)]== -9999
            num_of_rounds+=1
    if num_of_rounds == max_num_pruning_try:
        return Xsamples[random.randint(0, len(scores_copy)-1)]
        
#determine the number of sample configurations to skip           
def get_lookahead_status(model,delay):
    
    if len(exe_list)<delay:
        return 0
    else:
        lookahead_selection_list=[]
        j=0
        while j < lookahead_window:
            xx = opt_acquisition(XX, yy, model, acqval,1)
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
        

#define the BO models
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
            xx = opt_acquisition(XX, yy, model, acqval,1)
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
            xx = opt_acquisition(XX, yy, model, acqval,0)
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
            xx = opt_acquisition(XX, yy, model, acqval,1)
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
        
    with open('param_list_port.txt', 'w') as f:
   	 for item in param_list:
        	print >> f, item
    with open('exe_list_port.txt', 'w') as f:
   	 for item in exe_list:
        	print >> f, item
    with open('model_list_port.txt', 'w') as f:
   	 for item in mm:
        	print >> f, item
    with open('delay_list_port.txt', 'w') as f:
   	 for item in delay_list:
        	print >> f, item
    with open('lookahead_list_port.txt', 'w') as f:
    	for item in lookahead_list:
        	print >> f, item         
    with open('pruned_config_list_port.txt', 'w') as f:
    	for item in pruned_config_list:
        	print >> f, item


##reset hardware knobs
hyperthreading(1)
u_comm="likwid-setFrequencies -umax 3.0"
sp.check_output(shlex.split(u_comm), stderr=FNULL)
f_comm="likwid-setFrequencies -c S0 -f 2.2"
sp.check_output(shlex.split(f_comm), stderr=FNULL)

print (th_list)

