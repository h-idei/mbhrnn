C++ code of multimodal Bayesian homeostatic recurrent neural network model (PV-RNN)

1. Tested environment
    -OS: Ubuntu (20.04)
    -gcc/g++ version 11.2.0

2. Installation guide
    This code requires only gcc/g++.

3. Demo and Instructions
    
The file structure is below.

“network.hpp”: Network hyperparameter setting, neural network classes, and supplemental functions (used in both learning and error regression).
“network.hpp” defines some classes including
- "Output", "OutputNM", "PVRNNLayer", and "PVRNNTopLayer", which are used for learning
- "ER_Output", "ER_OutputNM", "ER_PVRNNLayer", and "ER_PVRNNTopLayer", which are used for error regression

Codes for learning
“learning.cpp”: Main code for running program (learning)
“learning_data/”: Target data of learning is placed here (e.g., target0000000.txt ~ )
“learning_model/”: Trained model (e.g., synaptic weights) are saved here
“learning_generation/”: Reproduced sequences (learning results) are saved here
“plot_learning.py”: Code for generating figure

Codes for error regression
"error_regression_allostasis.cpp" : Main code for running program (error regression)
“test_generation_allostasis/” : Reproduced sequences (results of error regression) are saved here
(The folder structure depends on the hyperparameter lists defined in “error_regression.cpp”)
“plot_er.py”: Code for generating figure

3-1. Learning experiment 
***It may take about 5[h] to train one neural network
    
    #Compilation and execution
    g++-11 -std=c++14 -O3 -fopenmp learning.cpp -o exe.learning
    
    ./exe.learning
    #Trained model and reproduced timeseries will be saved for each network module as 
    #"./learning_model/associative/weight_dd_*.txt" and "./learning_generation/out_extero/output_*_*.txt".
    
    #“plot_learning.py” can be used to generate figures (pdf files) of timeseries
    #saved in “./”.
    python3 plot_learning.py -s 0 -e 100000
    

3-2. Test experiment 
***It may take about 2.5[h] for 10 trials by one neural network
    #Compilation and execution
    #“error_regression_allostasis.cpp” reads a trained model from "./learning_model/"
    #Test trial run
    g++-11 -std=c++14 -O3 error_regression_allostasis.cpp -o exe.test
    ./exe.test
    #Timeseries for each trial will be saved in “test_generation_allostasis/fw_200/pw_10/itr_200/lr_0.1/”
    
    #“plot_er.py” can be used to generate figure (pdf file) of timeseries for each trial
    #saved in the same folder with generated timeseries
    python3 plot_er.py
    
