//  Predictive-coding-inspired Variational RNN
//  error_regression.cpp
//  Copyright © 2022 Hayato Idei. All rights reserved.
//
#include "network.hpp"
#include <sys/time.h>

//hyper-parameter used in error regression
#define TEST_SEQ_NUM 10 //sequence number of target data in error regression
#define ADAM1 0.9
#define ADAM2 0.999
#define MAX_TIME_LENGTH 2200 //max time length including future states
#define TIME_LENGTH 2000
#define POINT 25
#define CUE_STATE_STEP 5
#define CUE_STATE_STEP_KEEP 10
#define FOOD_CYCLE 100
#define NO_FOOD_STEP 25
#define GET_DISTANCE 0.2
double sigmoid(double x, double bias, double gain)
{
  return 1.0 / (1.0 + exp(-gain * (x-bias)));
}
//robot error regression
int main(void){ 
    
    vector<vector<vector<double> > > real_state_extero(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_extero_num, 0)));
    vector<vector<vector<double> > > real_state_proprio(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > real_state_intero(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_intero_num, 0)));
    vector<vector<vector<double> > > target(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_num, 0)));
    
    vector<vector<vector<double> > > target_extero(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_extero_num, 0)));
    vector<vector<vector<double> > > target_proprio(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > joint_angle(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > target_intero(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_intero_num, 0)));
    
    vector<vector<vector<double> > > pe_extero(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_extero_num, 0)));
    vector<vector<vector<double> > > pe_proprio(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > pe_intero(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_intero_num, 0)));
    vector<vector<vector<double> > > wkl_exteroceptive(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(exteroceptive_z_num, 0)));
    vector<vector<vector<double> > > wkl_proprioceptive(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(proprioceptive_z_num, 0)));
    vector<vector<vector<double> > > wkl_interoceptive(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(interoceptive_z_num, 0)));
    vector<vector<vector<double> > > wkl_neuromodulation(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(neuromodulation_z_num, 0)));
    vector<vector<vector<double> > > wkl_associative(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(associative_z_num, 0)));
    vector<vector<vector<double> > > wkl_executive(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(executive_z_num, 0)));
    
    vector<vector<vector<double> > > in_p_mu_executive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(executive_z_num, 0)));
    vector<vector<vector<double> > > in_p_sigma_executive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(executive_z_num, 0)));
    vector<vector<vector<double> > > a_mu_executive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(executive_z_num, 0)));
    vector<vector<vector<double> > > a_sigma_executive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(executive_z_num, 0)));
    vector<vector<vector<double> > > wkl_executive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(executive_z_num, 0)));
    
    vector<vector<vector<double> > > in_p_mu_neuromodulation_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(neuromodulation_z_num, 0)));
    vector<vector<vector<double> > > in_p_sigma_neuromodulation_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(neuromodulation_z_num, 0)));
    vector<vector<vector<double> > > a_mu_neuromodulation_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(neuromodulation_z_num, 0)));
    vector<vector<vector<double> > > a_sigma_neuromodulation_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(neuromodulation_z_num, 0)));
    vector<vector<vector<double> > > wkl_neuromodulation_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(neuromodulation_z_num, 0)));
    
    vector<vector<vector<double> > > in_p_mu_associative_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(associative_z_num, 0)));
    vector<vector<vector<double> > > in_p_sigma_associative_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(associative_z_num, 0)));
    vector<vector<vector<double> > > a_mu_associative_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(associative_z_num, 0)));
    vector<vector<vector<double> > > a_sigma_associative_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(associative_z_num, 0)));
    vector<vector<vector<double> > > wkl_associative_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(associative_z_num, 0)));
    
    vector<vector<vector<double> > > in_p_mu_exteroceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(exteroceptive_z_num, 0)));
    vector<vector<vector<double> > > in_p_sigma_exteroceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(exteroceptive_z_num, 0)));
    vector<vector<vector<double> > > a_mu_exteroceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(exteroceptive_z_num, 0)));
    vector<vector<vector<double> > > a_sigma_exteroceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(exteroceptive_z_num, 0)));
    vector<vector<vector<double> > > wkl_exteroceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(exteroceptive_z_num, 0)));
    
    vector<vector<vector<double> > > in_p_mu_proprioceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(proprioceptive_z_num, 0)));
    vector<vector<vector<double> > > in_p_sigma_proprioceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(proprioceptive_z_num, 0)));
    vector<vector<vector<double> > > a_mu_proprioceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(proprioceptive_z_num, 0)));
    vector<vector<vector<double> > > a_sigma_proprioceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(proprioceptive_z_num, 0)));
    vector<vector<vector<double> > > wkl_proprioceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(proprioceptive_z_num, 0)));
    
    vector<vector<vector<double> > > in_p_mu_interoceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(interoceptive_z_num, 0)));
    vector<vector<vector<double> > > in_p_sigma_interoceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(interoceptive_z_num, 0)));
    vector<vector<vector<double> > > a_mu_interoceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(interoceptive_z_num, 0)));
    vector<vector<vector<double> > > a_sigma_interoceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(interoceptive_z_num, 0)));
    vector<vector<vector<double> > > wkl_interoceptive_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(interoceptive_z_num, 0)));
    
    vector<vector<vector<double> > > output_extero_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_extero_num, 0)));
    vector<vector<vector<double> > > output_proprio_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > output_intero_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_intero_num, 0)));
    
    vector<vector<vector<double> > > sensory_sigma_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_num, 0)));
    
    vector<vector<vector<double> > > pe_extero_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_extero_num, 0)));
    vector<vector<vector<double> > > pe_proprio_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > pe_intero_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_intero_num, 0)));
    
    vector<vector<vector<double> > > fe_past_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(1, 0)));
    vector<vector<vector<double> > > fe_future_window_head(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(1, 0)));
    
    vector<vector<double> > points(POINT, vector<double>(2, 0));
    for(int i=0; i<POINT; ++i){
        int x = i%5;
        int y = i/5;
        points[i][0] = -0.8 + x*0.4;
        points[i][1] = 0.8 - y*0.4;
    }
    
    //create food position
    uniform_int_distribution<> rand25(0, POINT-1);
    vector<vector<vector<double> > > food_position_state_sequence(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(3, 0)));// x, y, state
    int sample_point = rand25(engine);
    for(int s=0; s<TEST_SEQ_NUM; ++s){
        for(int t=0; t<TIME_LENGTH; ++t){
            if(t%FOOD_CYCLE==0){
                sample_point = rand25(engine);
            }
            food_position_state_sequence[s][t][0] = points[sample_point][0];
            food_position_state_sequence[s][t][1] = points[sample_point][1];
        }
    }
    
    //set model
    ER_PVRNNTopLayer executive(TEST_SEQ_NUM, MAX_TIME_LENGTH, executive_d_num, executive_z_num, executive_W, executive_W1, executive_tau, "executive");
    ER_PVRNNLayer associative(TEST_SEQ_NUM, MAX_TIME_LENGTH, associative_d_num, executive_d_num, associative_z_num, associative_W, associative_W1, associative_tau, "associative");
    
    ER_PVRNNLayer neuromodulation(TEST_SEQ_NUM, MAX_TIME_LENGTH, neuromodulation_d_num, executive_d_num, neuromodulation_z_num, neuromodulation_W, neuromodulation_W1, neuromodulation_tau, "neuromodulation");
    
    ER_PVRNNLayer exteroceptive(TEST_SEQ_NUM, MAX_TIME_LENGTH, exteroceptive_d_num, associative_d_num, exteroceptive_z_num, exteroceptive_W, exteroceptive_W1, exteroceptive_tau, "exteroceptive");
    ER_PVRNNLayer proprioceptive(TEST_SEQ_NUM, MAX_TIME_LENGTH, proprioceptive_d_num, associative_d_num, proprioceptive_z_num, proprioceptive_W, proprioceptive_W1, proprioceptive_tau, "proprioceptive");
    ER_PVRNNLayer interoceptive(TEST_SEQ_NUM, MAX_TIME_LENGTH, interoceptive_d_num, associative_d_num, interoceptive_z_num, interoceptive_W, interoceptive_W1, interoceptive_tau, "interoceptive");
    
    ER_Output out_extero(TEST_SEQ_NUM, MAX_TIME_LENGTH, x_extero_num, exteroceptive_d_num, "out_extero");
    ER_Output out_proprio(TEST_SEQ_NUM, MAX_TIME_LENGTH, x_proprio_num, proprioceptive_d_num, "out_proprio");
    ER_Output out_intero(TEST_SEQ_NUM, MAX_TIME_LENGTH, x_intero_num, interoceptive_d_num, "out_intero");
    
    ER_OutputNM out_nm(TEST_SEQ_NUM, MAX_TIME_LENGTH, x_num, neuromodulation_d_num, x_nm_tau, "out_nm");
    
    //concatenated variables used in backprop
    vector<vector<vector<double> > > x_mean(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_num, 0)));
    vector<vector<vector<double> > > x_sigma_extero(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_extero_num, 0)));
    vector<vector<vector<double> > > x_sigma_proprio(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > x_sigma_intero(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(x_intero_num, 0)));
    
    vector<double> exte_prop_intero_tau(exteroceptive_d_num+proprioceptive_d_num+interoceptive_d_num, 0);
    vector<vector<double> > exte_prop_intero_weight_ld(exteroceptive_d_num+proprioceptive_d_num+interoceptive_d_num, vector<double>(associative_d_num, 0)); 
    vector<vector<vector<double> > > exte_prop_intero_grad_internal_state_d(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(exteroceptive_d_num+proprioceptive_d_num+interoceptive_d_num, 0)));
    
    vector<double> nm_ass_tau(neuromodulation_d_num+associative_d_num, 0);
    vector<vector<double> > nm_ass_weight_ld(neuromodulation_d_num+associative_d_num, vector<double>(executive_d_num, 0)); 
    vector<vector<vector<double> > > nm_ass_grad_internal_state_d(TEST_SEQ_NUM, vector<vector<double> >(MAX_TIME_LENGTH, vector<double>(neuromodulation_d_num+associative_d_num, 0)));
    
    //set variables related with sensory uncertainty
    normal_distribution<> dist_sn(0.0, sqrt(0.0001)); //sensory noise
    normal_distribution<> dist_normal(0.0, 1.0); //sensory noise caused by interoception
    
    vector<int> future_window_list = {200}; //time window in future
    vector<int> past_window_list ={10}; //time window in past
    vector<int> iteration_list = {200};
    vector<double> learning_rate_list = {0.1};
    int future_window_list_size = int(future_window_list.size());
    int past_window_list_size = int(past_window_list.size());
    int iteration_list_size = int(iteration_list.size());
    int learning_rate_list_size = int(learning_rate_list.size());
    for(int fw=0;fw<future_window_list_size;++fw){
        int future_window = future_window_list[fw];
    for(int pw=0;pw<past_window_list_size;++pw){
        int past_window = past_window_list[pw];
        for(int itr=0;itr<iteration_list_size;++itr){
            int iteration = iteration_list[itr];
            for(int lr=0;lr<learning_rate_list_size;++lr){
                double alpha = learning_rate_list[lr];
                //make save directory
                stringstream path_generation_executive;
                path_generation_executive << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/executive";
                string str_path_generation_executive = path_generation_executive.str();
                mkdir(str_path_generation_executive.c_str(), 0777);
                
                stringstream path_generation_associative;
                path_generation_associative << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/associative";
                string str_path_generation_associative = path_generation_associative.str();
                mkdir(str_path_generation_associative.c_str(), 0777);
                
                stringstream path_generation_neuromodulation;
                path_generation_neuromodulation << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/neuromodulation";
                string str_path_generation_neuromodulation = path_generation_neuromodulation.str();
                mkdir(str_path_generation_neuromodulation.c_str(), 0777);
                
                stringstream path_generation_exteroceptive;
                path_generation_exteroceptive << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/exteroceptive";
                string str_path_generation_exteroceptive = path_generation_exteroceptive.str();
                mkdir(str_path_generation_exteroceptive.c_str(), 0777);
                
                stringstream path_generation_proprioceptive;
                path_generation_proprioceptive << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/proprioceptive";
                string str_path_generation_proprioceptive = path_generation_proprioceptive.str();
                mkdir(str_path_generation_proprioceptive.c_str(), 0777);
                
                stringstream path_generation_interoceptive;
                path_generation_interoceptive << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/interoceptive";
                string str_path_generation_interoceptive = path_generation_interoceptive.str();
                mkdir(str_path_generation_interoceptive.c_str(), 0777);
                
                stringstream path_generation_out_extero;
                path_generation_out_extero << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/out_extero";
                string str_path_generation_out_extero = path_generation_out_extero.str();
                mkdir(str_path_generation_out_extero.c_str(), 0777);
                
                stringstream path_generation_out_proprio;
                path_generation_out_proprio << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/out_proprio";
                string str_path_generation_out_proprio = path_generation_out_proprio.str();
                mkdir(str_path_generation_out_proprio.c_str(), 0777);
                
                stringstream path_generation_out_intero;
                path_generation_out_intero << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/out_intero";
                string str_path_generation_out_intero = path_generation_out_intero.str();
                mkdir(str_path_generation_out_intero.c_str(), 0777);
                
                stringstream path_generation_out_nm;
                path_generation_out_nm << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/out_nm";
                string str_path_generation_out_nm = path_generation_out_nm.str();
                mkdir(str_path_generation_out_nm.c_str(), 0777);
                
                stringstream path_generation_food;
                path_generation_food << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha << "/food";
                string str_path_generation_food = path_generation_food.str();
                mkdir(str_path_generation_food.c_str(), 0777);
                
                stringstream path_generation_fe;
                path_generation_fe << "./test_generation_allostasis" << "/fw_" << future_window << "/pw_" << past_window << "/itr_" << iteration << "/lr_" << alpha;
                string str_path_generation_fe = path_generation_fe.str();
                
                double max_sigma_intero = sqrt(0.1);
                double cue_x = 0.0;
                double cue_y = 0.4;
                //error regression
                for(int s=0;s<TEST_SEQ_NUM;++s){
                    //measure time
                    struct timeval start, end;
                    gettimeofday(&start, NULL);
                    double interoceptive_state = 0.0; // initialize interoception 
                    double cue_state_x = -0.8;
                    double cue_state_y = -0.8;
                    int cue_flag = 1;
                    int cue_count = 0;
                    double cue_state_x_store = 0.0;
                    double cue_state_y_store = 0.0;
                    int hit_count = 0;
                    for(int ct=0;ct<TIME_LENGTH;++ct){//ct: current time step
                        printf("Future window size: %d, Past window size: %d, Iteration size: %d, Alpha: %lf, Sequence: %d, Time step: %d\n", future_window, past_window, iteration, alpha, s, ct);
                        int initial_regression;
                        (ct==0) ? (initial_regression=1) : (initial_regression=0);
                        int er_past_window;
                        (ct>=past_window-1) ? (er_past_window=past_window) : (er_past_window=ct+1);
                        int lr_normalization = er_past_window+future_window;
                        //int epoch_flag = 0;
                        for(int epoch=0;epoch<iteration;++epoch){
                            //Feedforward
                            for(int wt=ct-er_past_window+1;wt<=ct+future_window;++wt){//wt: time step within time window
                                int last_window_step=0;
                                (wt>=ct) ? (last_window_step=1) : (last_window_step=0);
                                
                                executive.er_forward(wt, s, epoch, initial_regression, last_window_step, "posterior");
                                associative.er_forward(executive.d, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                neuromodulation.er_forward(executive.d, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                exteroceptive.er_forward(associative.d, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                proprioceptive.er_forward(associative.d, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                interoceptive.er_forward(associative.d, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                
                                out_extero.er_forward(exteroceptive.d, wt, s);
                                out_proprio.er_forward(proprioceptive.d, wt, s);
                                out_intero.er_forward(interoceptive.d, wt, s);
                                out_nm.er_forward(neuromodulation.d, wt, s);
                                
                                //Set future target and so on
                                for(int i = 0; i < x_num; ++i){
                                    if(i<x_extero_num){
                                        x_mean[s][wt][i] = out_extero.output[s][wt][i];
                                        x_sigma_extero[s][wt][i] = out_nm.output[s][wt][i];
                                        if(wt>=ct+1){
                                            target_extero[s][wt][i] = out_extero.output[s][wt][i];
                                            target[s][wt][i] = x_mean[s][wt][i];
                                        }
                                    }else if(i<x_extero_num+x_proprio_num){
                                        x_mean[s][wt][i] = out_proprio.output[s][wt][i-x_extero_num];
                                        x_sigma_proprio[s][wt][i-x_extero_num] = out_nm.output[s][wt][i];
                                        if(wt>=ct+1){
                                            target_proprio[s][wt][i-x_extero_num] = out_proprio.output[s][wt][i-x_extero_num];
                                            target[s][wt][i] = x_mean[s][wt][i];
                                        }
                                    }else{
                                        x_mean[s][wt][i] = out_intero.output[s][wt][i-x_extero_num-x_proprio_num];
                                        x_sigma_intero[s][wt][i-x_extero_num-x_proprio_num] = out_nm.output[s][wt][i];
                                        if(wt>=ct+1){
                                            target_intero[s][wt][i-x_extero_num-x_proprio_num] = out_intero.output[s][wt][i-x_extero_num-x_proprio_num];
                                            target[s][wt][i] = x_mean[s][wt][i];
                                        }
                                    }
                                }
                            }
                            
                            
                            //set incoming target at the current time step (if online robot operation)
                            //set through PID control using proprioceptive prediction in robot operation
                            if(epoch==0){
                                double std = max_sigma_intero*(sigmoid(-interoceptive_state, 0.5, 20) + sigmoid(interoceptive_state, 0.5, 20));
                                if(ct%FOOD_CYCLE<=FOOD_CYCLE-NO_FOOD_STEP){
                                    food_position_state_sequence[s][ct][2] = 0.2*(1.0-1.0/(FOOD_CYCLE-NO_FOOD_STEP)*(ct%FOOD_CYCLE));
                                }else{
                                    food_position_state_sequence[s][ct][2] = 0.0;
                                }
                                real_state_extero[s][ct][0] = cue_state_x;
                                real_state_extero[s][ct][1] = cue_state_y;
                                target_extero[s][ct][0] = cue_state_x + std*dist_normal(engine) + dist_sn(engine);
                                target_extero[s][ct][1] = cue_state_y + std*dist_normal(engine) + dist_sn(engine);
                                for(int i=0; i<x_extero_num; ++i){
                                    if(target_extero[s][ct][i] >= 1.0){
                                        target_extero[s][ct][i] = 1.0;
                                    }else if(target_extero[s][ct][i] <= -1.0){
                                        target_extero[s][ct][i] = -1.0;
                                    }
                                    target[s][ct][i]=target_extero[s][ct][i];
                                }
                                
                                for(int i=0;i<x_proprio_num;++i){
                                    if(ct==0){
                                        joint_angle[s][ct][i] = PID(joint_angle[s][ct][i], out_proprio.output[s][ct][i]);
                                    }else{
                                        joint_angle[s][ct][i] = PID(joint_angle[s][ct-1][i], out_proprio.output[s][ct][i]);
                                    }
                                    real_state_proprio[s][ct][i] = joint_angle[s][ct][i];
                                    target_proprio[s][ct][i] = joint_angle[s][ct][i] + std*dist_normal(engine) + dist_sn(engine);
                                    if(target_proprio[s][ct][i] >= 1.0){
                                        target_proprio[s][ct][i] = 1.0;
                                    }else if(target_proprio[s][ct][i] <= -1.0){
                                        target_proprio[s][ct][i] = -1.0;
                                    }
                                    target[s][ct][i+x_extero_num] = target_proprio[s][ct][i];
                                }
                                real_state_intero[s][ct][0] = interoceptive_state;
                                target_intero[s][ct][0] = interoceptive_state + std*dist_normal(engine) + dist_sn(engine);
                                if(target_intero[s][ct][0] >= 1.0){
                                    target_intero[s][ct][0] = 1.0;
                                }else if(target_intero[s][ct][0] <= -1.0){
                                    target_intero[s][ct][0] = -1.0;
                                }
                                target[s][ct][x_extero_num+x_proprio_num] = target_intero[s][ct][0];
                                
                                double food_distance = sqrt(pow(real_state_proprio[s][ct][0] - food_position_state_sequence[s][ct][0],2) + pow(real_state_proprio[s][ct][1] - food_position_state_sequence[s][ct][1],2));
                                double cue_distance = sqrt(pow(real_state_proprio[s][ct][0] - cue_x,2) + pow(real_state_proprio[s][ct][1] - cue_y,2));
                                double movement;
                                if(ct==0){
                                    movement = 0.0;
                                }else{
                                    movement = sqrt(pow(real_state_proprio[s][ct][0] - real_state_proprio[s][ct-1][0],2) + pow(real_state_proprio[s][ct][1] - real_state_proprio[s][ct-1][1],2));
                                }
                                interoceptive_state -= 0.013*(0.1+movement);
                                
                                //get food
                                if(food_distance <= GET_DISTANCE){
                                    interoceptive_state += food_position_state_sequence[s][ct][2];
                                    hit_count++;
                                }
                                //get cue
                                if(ct%FOOD_CYCLE>=FOOD_CYCLE-5){
                                    if(ct%FOOD_CYCLE==FOOD_CYCLE-5){
                                        cue_state_x_store = cue_state_x;
                                        cue_state_y_store = cue_state_y;
                                    }
                                    cue_state_x -= (cue_state_x_store+0.8)/5;
                                    cue_state_y -= (cue_state_y_store+0.8)/5;
                                    cue_flag = 1;
                                    cue_count = 0;
                                }else{
                                    if(cue_flag==1){
                                        if(cue_distance <= GET_DISTANCE){
                                            cue_flag = 0;                 
                                        }
                                    }else if(cue_flag==0){
                                        cue_count++;
                                        if(cue_count<=CUE_STATE_STEP){
                                            cue_state_x += (food_position_state_sequence[s][ct][0]+0.8)/CUE_STATE_STEP;
                                            cue_state_y += (food_position_state_sequence[s][ct][1]+0.8)/CUE_STATE_STEP;
                                        }else if(cue_count>CUE_STATE_STEP && cue_count<=CUE_STATE_STEP+CUE_STATE_STEP_KEEP){
                                            cue_state_x = food_position_state_sequence[s][ct][0];
                                            cue_state_y = food_position_state_sequence[s][ct][1];
                                        }else if(cue_count>CUE_STATE_STEP+CUE_STATE_STEP_KEEP && cue_count<=2*CUE_STATE_STEP+CUE_STATE_STEP_KEEP){
                                            cue_state_x -= (food_position_state_sequence[s][ct][0]+0.8)/CUE_STATE_STEP;
                                            cue_state_y -= (food_position_state_sequence[s][ct][1]+0.8)/CUE_STATE_STEP;
                                        }else{
                                            cue_flag = 1;
                                            cue_count = 0;
                                        }  
                                    }
                                }
                                if(interoceptive_state >= 1.0){
                                    interoceptive_state = 1.0;
                                }else if(interoceptive_state <= -1.0){
                                    interoceptive_state = -1.0;
                                }     
                            }
                            
                            //error
                            double normalize = 1.0/lr_normalization;
                            fe_past_window_head[s][ct][0] = 0.0;
                            fe_future_window_head[s][ct][0] = 0.0;
                            for(int wt=ct-er_past_window+1;wt<=ct+future_window;++wt){
                                double future_entropy_constant_term;
                                (wt<=ct) ? (future_entropy_constant_term=0.0) : (future_entropy_constant_term=1.0);
                                double past_flag;
                                (wt<=ct) ? (past_flag=1.0) : (past_flag=0.0);
                                double future_flag;
                                (wt<=ct) ? (future_flag=0.0) : (future_flag=1.0);
                                for(int i=0; i<x_extero_num; ++i){
                                    pe_extero[s][wt][i] = 0.5*(pow((target_extero[s][wt][i]-out_extero.output[s][wt][i])/x_sigma_extero[s][wt][i],2)+log(2*PI)+2.0*log(x_sigma_extero[s][wt][i])+future_entropy_constant_term)*normalize/x_extero_num;
                                    fe_past_window_head[s][ct][0] += past_flag*pe_extero[s][wt][i];
                                    fe_future_window_head[s][ct][0] += future_flag*pe_extero[s][wt][i];
                                }
                                for(int i=0; i<x_proprio_num; ++i){
                                    pe_proprio[s][wt][i] = 0.5*(pow((target_proprio[s][wt][i]-out_proprio.output[s][wt][i])/x_sigma_proprio[s][wt][i],2)+log(2*PI)+2.0*log(x_sigma_proprio[s][wt][i])+future_entropy_constant_term)*normalize/x_proprio_num;
                                    fe_past_window_head[s][ct][0] += past_flag*pe_proprio[s][wt][i];
                                    fe_future_window_head[s][ct][0] += future_flag*pe_proprio[s][wt][i];
                                }
                                for(int i=0; i<x_intero_num; ++i){
                                    pe_intero[s][wt][i] = 0.5*(pow((target_intero[s][wt][i]-out_intero.output[s][wt][i])/x_sigma_intero[s][wt][i],2)+log(2*PI)+2.0*log(x_sigma_intero[s][wt][i])+future_entropy_constant_term)*normalize/x_intero_num;
                                    fe_past_window_head[s][ct][0] += past_flag*pe_intero[s][wt][i];
                                    fe_future_window_head[s][ct][0] += future_flag*pe_intero[s][wt][i];
                                }
                                for(int i=0; i<exteroceptive_z_num; ++i){
                                    wkl_exteroceptive[s][wt][i] = exteroceptive_W*(log(exteroceptive.p_sigma[s][wt][i])-log(exteroceptive.q_sigma[s][wt][i])+0.5*(pow(exteroceptive.p_mu[s][wt][i]-exteroceptive.q_mu[s][wt][i],2)+pow(exteroceptive.q_sigma[s][wt][i],2))/pow(exteroceptive.p_sigma[s][wt][i],2)-0.5)*normalize/exteroceptive_z_num;
                                    fe_past_window_head[s][ct][0] += past_flag*wkl_exteroceptive[s][wt][i];
                                    fe_future_window_head[s][ct][0] += future_flag*wkl_exteroceptive[s][wt][i];
                                }
                                for(int i=0; i<proprioceptive_z_num; ++i){
                                    wkl_proprioceptive[s][wt][i] = proprioceptive_W*(log(proprioceptive.p_sigma[s][wt][i])-log(proprioceptive.q_sigma[s][wt][i])+0.5*(pow(proprioceptive.p_mu[s][wt][i]-proprioceptive.q_mu[s][wt][i],2)+pow(proprioceptive.q_sigma[s][wt][i],2))/pow(proprioceptive.p_sigma[s][wt][i],2)-0.5)*normalize/proprioceptive_z_num;
                                    fe_past_window_head[s][ct][0] += past_flag*wkl_proprioceptive[s][wt][i];
                                    fe_future_window_head[s][ct][0] += future_flag*wkl_proprioceptive[s][wt][i];
                                }
                                for(int i=0; i<interoceptive_z_num; ++i){
                                    wkl_interoceptive[s][wt][i] = interoceptive_W*(log(interoceptive.p_sigma[s][wt][i])-log(interoceptive.q_sigma[s][wt][i])+0.5*(pow(interoceptive.p_mu[s][wt][i]-interoceptive.q_mu[s][wt][i],2)+pow(interoceptive.q_sigma[s][wt][i],2))/pow(interoceptive.p_sigma[s][wt][i],2)-0.5)*normalize/interoceptive_z_num;
                                    fe_past_window_head[s][ct][0] += past_flag*wkl_interoceptive[s][wt][i];
                                    fe_future_window_head[s][ct][0] += future_flag*wkl_interoceptive[s][wt][i];
                                }
                                for(int i=0; i<neuromodulation_z_num; ++i){
                                    wkl_neuromodulation[s][wt][i] = neuromodulation_W*(log(neuromodulation.p_sigma[s][wt][i])-log(neuromodulation.q_sigma[s][wt][i])+0.5*(pow(neuromodulation.p_mu[s][wt][i]-neuromodulation.q_mu[s][wt][i],2)+pow(neuromodulation.q_sigma[s][wt][i],2))/pow(neuromodulation.p_sigma[s][wt][i],2)-0.5)*normalize/neuromodulation_z_num;
                                    fe_past_window_head[s][ct][0] += past_flag*wkl_neuromodulation[s][wt][i];
                                    fe_future_window_head[s][ct][0] += future_flag*wkl_neuromodulation[s][wt][i];
                                }
                                for(int i=0; i<associative_z_num; ++i){
                                    wkl_associative[s][wt][i] = associative_W*(log(associative.p_sigma[s][wt][i])-log(associative.q_sigma[s][wt][i])+0.5*(pow(associative.p_mu[s][wt][i]-associative.q_mu[s][wt][i],2)+pow(associative.q_sigma[s][wt][i],2))/pow(associative.p_sigma[s][wt][i],2)-0.5)*normalize/associative_z_num;
                                    fe_past_window_head[s][ct][0] += past_flag*wkl_associative[s][wt][i];
                                    fe_future_window_head[s][ct][0] += future_flag*wkl_associative[s][wt][i];
                                }
                                for(int i=0; i<executive_z_num; ++i){
                                    wkl_executive[s][wt][i] = executive_W*(log(executive.p_sigma[s][wt][i])-log(executive.q_sigma[s][wt][i])+0.5*(pow(executive.p_mu[s][wt][i]-executive.q_mu[s][wt][i],2)+pow(executive.q_sigma[s][wt][i],2))/pow(executive.p_sigma[s][wt][i],2)-0.5)*normalize/executive_z_num;
                                    fe_past_window_head[s][ct][0] += past_flag*wkl_executive[s][wt][i];
                                    fe_future_window_head[s][ct][0] += future_flag*wkl_executive[s][wt][i];
                                }
                            }
                            
                            //Backward
                            //set constant for Rectified Adam
                            double rho_inf = 2.0/(1.0-ADAM2)-1.0;
                            double rho = rho_inf-2.0*(epoch+1)*pow(ADAM2,epoch+1)/(1.0-pow(ADAM2,epoch+1));
                            double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/ (rho_inf-4.0)/(rho_inf-2.0)/rho);
                            double m_radam1 = 1.0/(1.0-pow(ADAM1,epoch+1));
                            double l_radam2 = 1.0-pow(ADAM2,epoch+1);
                            for(int wt=ct+future_window;wt>=ct-er_past_window+1;--wt){
                                int last_window_step;
                                (wt==ct+future_window) ? (last_window_step=1) : (last_window_step=0);
                                out_intero.er_backward_nm(target_intero, x_sigma_intero, wt, s, lr_normalization);
                                out_proprio.er_backward_nm(target_proprio, x_sigma_proprio, wt, s, lr_normalization);
                                out_extero.er_backward_nm(target_extero, x_sigma_extero, wt, s, lr_normalization);
                                out_nm.er_backward(target, x_mean, last_window_step, wt, s, lr_normalization);
                                
                                interoceptive.er_backward(out_intero.weight_ohd, out_intero.grad_internal_state_output, out_intero.tau, last_window_step, wt, s, lr_normalization);
                                proprioceptive.er_backward(out_proprio.weight_ohd, out_proprio.grad_internal_state_output, out_proprio.tau, last_window_step, wt, s, lr_normalization);
                                exteroceptive.er_backward(out_extero.weight_ohd, out_extero.grad_internal_state_output, out_extero.tau, last_window_step, wt, s, lr_normalization);
                                neuromodulation.er_backward(out_nm.weight_ohd, out_nm.grad_internal_state_output, out_nm.tau, last_window_step, wt, s, lr_normalization);
                                
                                //concatenate some variables in proprioceptive and exteroceptive areas to make backprop-inputs to the associative area
                                for(int i=0;i<exteroceptive_d_num+proprioceptive_d_num+interoceptive_d_num;++i){
                                    if(i<exteroceptive_d_num) {
                                        exte_prop_intero_grad_internal_state_d[s][wt][i] = exteroceptive.grad_internal_state_d[s][wt][i];
                                        exte_prop_intero_tau[i] = exteroceptive.tau[i];
                                    }else if(i<exteroceptive_d_num+proprioceptive_d_num){
                                        exte_prop_intero_grad_internal_state_d[s][wt][i] = proprioceptive.grad_internal_state_d[s][wt][i-exteroceptive_d_num];
                                        exte_prop_intero_tau[i] = proprioceptive.tau[i-exteroceptive_d_num];
                                    }else{
                                        exte_prop_intero_grad_internal_state_d[s][wt][i] = interoceptive.grad_internal_state_d[s][wt][i-exteroceptive_d_num-proprioceptive_d_num];
                                        exte_prop_intero_tau[i] = interoceptive.tau[i-exteroceptive_d_num-proprioceptive_d_num];
                                    }
                                    for(int j=0;j<associative_d_num;++j){
                                        if(i<exteroceptive_d_num){
                                            exte_prop_intero_weight_ld[i][j] = exteroceptive.weight_dhd[i][j];
                                        }else if(i<exteroceptive_d_num+proprioceptive_d_num){
                                            exte_prop_intero_weight_ld[i][j] = proprioceptive.weight_dhd[i-exteroceptive_d_num][j];
                                        }else{
                                            exte_prop_intero_weight_ld[i][j] = interoceptive.weight_dhd[i-exteroceptive_d_num-proprioceptive_d_num][j];
                                        }
                                    }
                                }
                                associative.er_backward(exte_prop_intero_weight_ld, exte_prop_intero_grad_internal_state_d, exte_prop_intero_tau, last_window_step, wt, s, lr_normalization);
                                
                                //concatenate some variables in associative and neuromodulation areas to make backprop-inputs to the executive area
                                for(int i=0;i<neuromodulation_d_num+associative_d_num;++i){
                                    if(i<neuromodulation_d_num){
                                        nm_ass_grad_internal_state_d[s][wt][i] = neuromodulation.grad_internal_state_d[s][wt][i];
                                        nm_ass_tau[i] = neuromodulation.tau[i];
                                    }else{
                                        nm_ass_grad_internal_state_d[s][wt][i] = associative.grad_internal_state_d[s][wt][i-neuromodulation_d_num];
                                        nm_ass_tau[i] = associative.tau[i-neuromodulation_d_num];
                                    }
                                    for(int j=0;j<executive_d_num;++j){
                                        if(i<neuromodulation_d_num){
                                            nm_ass_weight_ld[i][j] = neuromodulation.weight_dhd[i][j];
                                        }else{
                                            nm_ass_weight_ld[i][j] = associative.weight_dhd[i-neuromodulation_d_num][j];
                                        }
                                    }
                                }
                
                                executive.er_backward(nm_ass_weight_ld, nm_ass_grad_internal_state_d, nm_ass_tau, last_window_step, wt, s, lr_normalization);
                                
                                //update adaptive vector (Rectified Adam)
                                interoceptive.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                proprioceptive.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                exteroceptive.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                neuromodulation.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                associative.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                executive.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                            }
                            
                        }
                        //set timeseries at the head of the past time window
                        for(int i=0; i<executive_z_num; ++i){
                            in_p_mu_executive_window_head[s][ct][i] = executive.internal_state_p_mu[s][ct][i];
                            in_p_sigma_executive_window_head[s][ct][i] = executive.internal_state_p_sigma[s][ct][i];
                            a_mu_executive_window_head[s][ct][i] = executive.a_mu[s][ct][i];
                            a_sigma_executive_window_head[s][ct][i] = executive.a_sigma[s][ct][i];
                            wkl_executive_window_head[s][ct][i] = wkl_executive[s][ct][i];
                        }
                        for(int i=0; i<neuromodulation_z_num; ++i){
                            in_p_mu_neuromodulation_window_head[s][ct][i] = neuromodulation.internal_state_p_mu[s][ct][i];
                            in_p_sigma_neuromodulation_window_head[s][ct][i] = neuromodulation.internal_state_p_sigma[s][ct][i];
                            a_mu_neuromodulation_window_head[s][ct][i] = neuromodulation.a_mu[s][ct][i];
                            a_sigma_neuromodulation_window_head[s][ct][i] = neuromodulation.a_sigma[s][ct][i];
                            wkl_neuromodulation_window_head[s][ct][i] = wkl_neuromodulation[s][ct][i];
                        }
                        for(int i=0; i<associative_z_num; ++i){
                            in_p_mu_associative_window_head[s][ct][i] = associative.internal_state_p_mu[s][ct][i];
                            in_p_sigma_associative_window_head[s][ct][i] = associative.internal_state_p_sigma[s][ct][i];
                            a_mu_associative_window_head[s][ct][i] = associative.a_mu[s][ct][i];
                            a_sigma_associative_window_head[s][ct][i] = associative.a_sigma[s][ct][i];
                            wkl_associative_window_head[s][ct][i] = wkl_associative[s][ct][i];
                        }
                        for(int i=0; i<exteroceptive_z_num; ++i){
                            in_p_mu_exteroceptive_window_head[s][ct][i] = exteroceptive.internal_state_p_mu[s][ct][i];
                            in_p_sigma_exteroceptive_window_head[s][ct][i] = exteroceptive.internal_state_p_sigma[s][ct][i];
                            a_mu_exteroceptive_window_head[s][ct][i] = exteroceptive.a_mu[s][ct][i];
                            a_sigma_exteroceptive_window_head[s][ct][i] = exteroceptive.a_sigma[s][ct][i];
                            wkl_exteroceptive_window_head[s][ct][i] = wkl_exteroceptive[s][ct][i];
                        }
                        for(int i=0; i<proprioceptive_z_num; ++i){
                            in_p_mu_proprioceptive_window_head[s][ct][i] = proprioceptive.internal_state_p_mu[s][ct][i];
                            in_p_sigma_proprioceptive_window_head[s][ct][i] = proprioceptive.internal_state_p_sigma[s][ct][i];
                            a_mu_proprioceptive_window_head[s][ct][i] = proprioceptive.a_mu[s][ct][i];
                            a_sigma_proprioceptive_window_head[s][ct][i] = proprioceptive.a_sigma[s][ct][i];
                            wkl_proprioceptive_window_head[s][ct][i] = wkl_proprioceptive[s][ct][i];
                        }
                        for(int i=0; i<interoceptive_z_num; ++i){
                            in_p_mu_interoceptive_window_head[s][ct][i] = interoceptive.internal_state_p_mu[s][ct][i];
                            in_p_sigma_interoceptive_window_head[s][ct][i] = interoceptive.internal_state_p_sigma[s][ct][i];
                            a_mu_interoceptive_window_head[s][ct][i] = interoceptive.a_mu[s][ct][i];
                            a_sigma_interoceptive_window_head[s][ct][i] = interoceptive.a_sigma[s][ct][i];
                            wkl_interoceptive_window_head[s][ct][i] = wkl_interoceptive[s][ct][i];
                        }
                        for(int i = 0; i < x_num; ++i){
                            sensory_sigma_window_head[s][ct][i] = out_nm.output[s][ct][i];
                            if(i<x_extero_num){
                                output_extero_window_head[s][ct][i] = out_extero.output[s][ct][i];
                                pe_extero_window_head[s][ct][i] = pe_extero[s][ct][i];
                            }else if(i<x_extero_num+x_proprio_num){
                                output_proprio_window_head[s][ct][i-x_extero_num] = out_proprio.output[s][ct][i-x_extero_num];
                                pe_proprio_window_head[s][ct][i-x_extero_num] = pe_proprio[s][ct][i-x_extero_num];
                            }else{
                                output_intero_window_head[s][ct][i-x_extero_num-x_proprio_num] = out_intero.output[s][ct][i-x_extero_num-x_proprio_num];
                                pe_intero_window_head[s][ct][i-x_extero_num-x_proprio_num] = pe_intero[s][ct][i-x_extero_num-x_proprio_num];
                            }
                        }
                        
                        //Save generated sequence
                        if(ct==TIME_LENGTH-1){
                        executive.er_save_sequence(str_path_generation_executive, TIME_LENGTH, s, ct);
                        associative.er_save_sequence(str_path_generation_associative, TIME_LENGTH, s, ct);
                        neuromodulation.er_save_sequence(str_path_generation_neuromodulation, TIME_LENGTH, s, ct);
                        exteroceptive.er_save_sequence(str_path_generation_exteroceptive, TIME_LENGTH, s, ct);
                        proprioceptive.er_save_sequence(str_path_generation_proprioceptive, TIME_LENGTH, s, ct);
                        interoceptive.er_save_sequence(str_path_generation_interoceptive, TIME_LENGTH, s, ct);
                        out_nm.er_save_sequence(str_path_generation_out_nm, TIME_LENGTH, s, ct);
                        out_extero.er_save_sequence(str_path_generation_out_extero, target_extero, TIME_LENGTH, s, ct);
                        out_proprio.er_save_sequence(str_path_generation_out_proprio, target_proprio, TIME_LENGTH, s, ct);
                        out_intero.er_save_sequence(str_path_generation_out_intero, target_intero, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_executive, "wkld", wkl_executive, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_associative, "wkld", wkl_associative, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_neuromodulation, "wkld", wkl_neuromodulation, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_exteroceptive, "wkld", wkl_exteroceptive, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_proprioceptive, "wkld", wkl_proprioceptive, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_interoceptive, "wkld", wkl_interoceptive, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_out_extero, "pe", pe_extero, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_out_proprio, "pe", pe_proprio, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_out_intero, "pe", pe_intero, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_out_extero, "real", real_state_extero, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_out_proprio, "real", real_state_proprio, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_out_intero, "real", real_state_intero, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_food, "food", food_position_state_sequence, TIME_LENGTH, s, ct);
                        
                        //save timeseries at the head of the past time window
                        save_generated_sequence(str_path_generation_executive, "in_p_mu_window_head", in_p_mu_executive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_executive, "in_p_sigma_window_head", in_p_sigma_executive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_executive, "a_mu_window_head", a_mu_executive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_executive, "a_sigma_window_head", a_sigma_executive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_executive, "wkld_window_head", wkl_executive_window_head, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_neuromodulation, "in_p_mu_window_head", in_p_mu_neuromodulation_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_neuromodulation, "in_p_sigma_window_head", in_p_sigma_neuromodulation_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_neuromodulation, "a_mu_window_head", a_mu_neuromodulation_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_neuromodulation, "a_sigma_window_head", a_sigma_neuromodulation_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_neuromodulation, "wkld_window_head", wkl_neuromodulation_window_head, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_associative, "in_p_mu_window_head", in_p_mu_associative_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_associative, "in_p_sigma_window_head", in_p_sigma_associative_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_associative, "a_mu_window_head", a_mu_associative_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_associative, "a_sigma_window_head", a_sigma_associative_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_associative, "wkld_window_head", wkl_associative_window_head, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_exteroceptive, "in_p_mu_window_head", in_p_mu_exteroceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_exteroceptive, "in_p_sigma_window_head", in_p_sigma_exteroceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_exteroceptive, "a_mu_window_head", a_mu_exteroceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_exteroceptive, "a_sigma_window_head", a_sigma_exteroceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_exteroceptive, "wkld_window_head", wkl_exteroceptive_window_head, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_proprioceptive, "in_p_mu_window_head", in_p_mu_proprioceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_proprioceptive, "in_p_sigma_window_head", in_p_sigma_proprioceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_proprioceptive, "a_mu_window_head", a_mu_proprioceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_proprioceptive, "a_sigma_window_head", a_sigma_proprioceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_proprioceptive, "wkld_window_head", wkl_proprioceptive_window_head, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_interoceptive, "in_p_mu_window_head", in_p_mu_interoceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_interoceptive, "in_p_sigma_window_head", in_p_sigma_interoceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_interoceptive, "a_mu_window_head", a_mu_interoceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_interoceptive, "a_sigma_window_head", a_sigma_interoceptive_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_interoceptive, "wkld_window_head", wkl_interoceptive_window_head, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_out_extero, "output_window_head", output_extero_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_out_proprio, "output_window_head", output_proprio_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_out_intero, "output_window_head", output_intero_window_head, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_out_nm, "output_window_head", sensory_sigma_window_head, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_out_extero, "pe_window_head", pe_extero_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_out_proprio, "pe_window_head", pe_proprio_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_out_intero, "pe_window_head", pe_intero_window_head, TIME_LENGTH, s, ct);
                        
                        save_generated_sequence(str_path_generation_fe, "fe_past_window_head", fe_past_window_head, TIME_LENGTH, s, ct);
                        save_generated_sequence(str_path_generation_fe, "fe_future_window_head", fe_future_window_head, TIME_LENGTH, s, ct);
                        }
                    }
                    
                    printf("Final Interoceptive state: %lf \n", interoceptive_state);
                    
                    gettimeofday(&end, NULL);
                    float delta = end.tv_sec  - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
                    printf("time %lf[s]\n", delta);
                }
            }
        }
    }
    }
    return 0;
}
