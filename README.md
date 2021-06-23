# Bliss: Auto-tuning Complex Applications using a Pool of Diverse Lightweight Learning Models 

## Dependencies:
Please install the following library dependencies in Python3.6 to run BLISS. <\br>
```
pip install numpy 
pip install scipy
pip install scikit-optimize
```
BLISS is tested for [Kripke](https://github.com/LLNL/Kripke), [Clomp](https://asc.llnl.gov/coral-2-benchmarks), [AMG](https://github.com/LLNL/AMG), [Hypre](https://github.com/LLNL/AMG) and [Lulesh](https://asc.llnl.gov/codes/proxy-apps/lulesh). Install the applications from their source. </br> 

## Run BLISS

BLISS can be run both *with and without portability aid*. </br>
To run BLISS without portability aid, run ``` sudo python3 ./setup/main.py ``` </br>
To run BLISS with portability aid, ``` sudo python3 ./setup/main_portabe.py ``` </br>
Note that, to run BLISS, define the search space, application binary path, and also ensure that your system supports tuning of hardware parameters like hyperthreading, uncore frequency and core frequency. If your system does not support changing these hardware parameters, run BLISS with only software parameters and accordingly define the search space. </br>
BLISS generates the following major files as output data: </br>
(1) *exe_list.txt:* It contains the execution time of all the configurations sampled. </br>
(2) *lookahead_list.txt:* It contains the the information about how many sample evaluations were skipped by BLISS. Their values are predicted from BLISS' surrogate model. </br>
(3) *delay_list.txt:* It contains the number of sampling done before BLISS decides attains maturity.</br>
(4) *model_list.txt:* It contains the BO models chosen by BLISS in each sample evaluation.</br>
(5) *param_list.txt:* It contains the parameter configuration chosen in each sample evaluation.</br> 
