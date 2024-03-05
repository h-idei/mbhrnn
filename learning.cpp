//  Predictive-coding-inspired Variational RNN
//  learning.cpp
//  Copyright © 2022 Hayato Idei. All rights reserved.
//
#include "network.hpp"
#include <sys/time.h>

//hyper-parameter for learning
#define EPOCH 100000 // iteration of parameter update in training (set in the main function in error regression) 
#define SAVE_EPOCH 10000
#define SEQ_NUM 1 //sequence number of target data in training or error regression in simulation
#define MAX_LENGTH 10000 //used for initializing shape of variables 
#define ALPHA 0.01 //learning rate
#define WEIGHT_DECAY 0.0005
#define ADAM1 0.9
#define ADAM2 0.999
double sigmoid(double x, double bias, double gain)
{
  return 1.0 / (1.0 + exp(-gain * (x-bias)));
}
//example of learning
int main(void){ 
    
    //set path to output file
    string fo_path_param_executive = "./learning_model/executive";
    string fo_path_generation_executive = "./learning_generation/executive";
    
    string fo_path_param_associative = "./learning_model/associative";
    string fo_path_generation_associative = "./learning_generation/associative";
    
    string fo_path_param_neuromodulation = "./learning_model/neuromodulation";
    string fo_path_generation_neuromodulation = "./learning_generation/neuromodulation";
    
    string fo_path_param_exteroceptive = "./learning_model/exteroceptive";
    string fo_path_generation_exteroceptive = "./learning_generation/exteroceptive";
    
    string fo_path_param_proprioceptive = "./learning_model/proprioceptive";
    string fo_path_generation_proprioceptive = "./learning_generation/proprioceptive";
    
    string fo_path_param_interoceptive = "./learning_model/interoceptive";
    string fo_path_generation_interoceptive = "./learning_generation/interoceptive";
    
    string fo_path_param_out_extero = "./learning_model/out_extero";
    string fo_path_generation_out_extero = "./learning_generation/out_extero";
    string fo_path_param_out_proprio = "./learning_model/out_proprio";
    string fo_path_generation_out_proprio = "./learning_generation/out_proprio";
    string fo_path_param_out_intero = "./learning_model/out_intero";
    string fo_path_generation_out_intero = "./learning_generation/out_intero";
    string fo_path_param_out_nm = "./learning_model/out_nm";
    string fo_path_generation_out_nm = "./learning_generation/out_nm";
    
    //set target data
    vector<int> length(SEQ_NUM, 0);
    vector<vector<vector<double> > > target_external(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_num, 0)));
    set_target_data(length, target_external, "learning_data");
    
    vector<vector<vector<double> > > target_extero(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_extero_num, 0)));
    vector<vector<vector<double> > > target_proprio(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > target_intero(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_intero_num, 0)));
    for(int s=0;s<SEQ_NUM;++s){
        for(int t=0;t<MAX_LENGTH;++t){
            for (int i = 0; i < x_extero_num; ++i) {
                target_extero[s][t][i] = target_external[s][t][i];
            }
            for (int i = 0; i < x_proprio_num; ++i) {
                target_proprio[s][t][i] = target_external[s][t][i+x_extero_num];
            }
            for (int i = 0; i < x_intero_num; ++i) {
                target_intero[s][t][i] = target_external[s][t][i+x_extero_num+x_proprio_num];
            }
        }
    }
    
    //set model
    PVRNNTopLayer executive(SEQ_NUM, MAX_LENGTH, executive_d_num, executive_z_num, executive_W, executive_W1, executive_tau, fo_path_param_executive, fo_path_generation_executive, "learning");
    PVRNNLayer associative(SEQ_NUM, MAX_LENGTH, associative_d_num, executive_d_num, associative_z_num, associative_W, associative_W1, associative_tau, fo_path_param_associative, fo_path_generation_associative, "learning");
    
    PVRNNLayer neuromodulation(SEQ_NUM, MAX_LENGTH, neuromodulation_d_num, executive_d_num, neuromodulation_z_num, neuromodulation_W, neuromodulation_W1, neuromodulation_tau, fo_path_param_neuromodulation, fo_path_generation_neuromodulation, "learning");
    
    PVRNNLayer exteroceptive(SEQ_NUM, MAX_LENGTH, exteroceptive_d_num, associative_d_num, exteroceptive_z_num, exteroceptive_W, exteroceptive_W1, exteroceptive_tau, fo_path_param_exteroceptive, fo_path_generation_exteroceptive, "learning");
    
    PVRNNLayer proprioceptive(SEQ_NUM, MAX_LENGTH, proprioceptive_d_num, associative_d_num, proprioceptive_z_num, proprioceptive_W, proprioceptive_W1, proprioceptive_tau, fo_path_param_proprioceptive, fo_path_generation_proprioceptive, "learning");
    PVRNNLayer interoceptive(SEQ_NUM, MAX_LENGTH, interoceptive_d_num, associative_d_num, interoceptive_z_num, interoceptive_W, interoceptive_W1, interoceptive_tau, fo_path_param_interoceptive, fo_path_generation_interoceptive, "learning");
    
    Output out_extero(SEQ_NUM, MAX_LENGTH, x_extero_num, exteroceptive_d_num, fo_path_param_out_extero, fo_path_generation_out_extero, "learning");
    Output out_proprio(SEQ_NUM, MAX_LENGTH, x_proprio_num, proprioceptive_d_num, fo_path_param_out_proprio, fo_path_generation_out_proprio, "learning");
    Output out_intero(SEQ_NUM, MAX_LENGTH, x_intero_num, interoceptive_d_num, fo_path_param_out_intero, fo_path_generation_out_intero, "learning");
    
    OutputNM out_nm(SEQ_NUM, MAX_LENGTH, x_num, neuromodulation_d_num, x_nm_tau, fo_path_param_out_nm, fo_path_generation_out_nm, "learning");
    
    //concatenated variables used in backprop
    vector<vector<vector<double> > > x_mean(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_num, 0)));
    vector<vector<vector<double> > > x_sigma_extero(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_extero_num, 0)));
    vector<vector<vector<double> > > x_sigma_proprio(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > x_sigma_intero(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_intero_num, 0)));
    
    vector<double> exte_prop_intero_tau(exteroceptive_d_num+proprioceptive_d_num+interoceptive_d_num, 0);
    vector<vector<double> > exte_prop_intero_weight_ld(exteroceptive_d_num+proprioceptive_d_num+interoceptive_d_num, vector<double>(associative_d_num, 0)); 
    vector<vector<vector<double> > > exte_prop_intero_grad_internal_state_d(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(exteroceptive_d_num+proprioceptive_d_num+interoceptive_d_num, 0)));
    
    vector<double> nm_ass_tau(neuromodulation_d_num+associative_d_num, 0);
    vector<vector<double> > nm_ass_weight_ld(neuromodulation_d_num+associative_d_num, vector<double>(executive_d_num, 0)); 
    vector<vector<vector<double> > > nm_ass_grad_internal_state_d(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(neuromodulation_d_num+associative_d_num, 0)));
    
    //measure time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    //PVRNN learning
    //reset loss (free-energy)
    vector<double> loss_seq(SEQ_NUM, 0);
    double loss=0;
    
    for(int epoch=0;epoch<EPOCH;++epoch){
        //ramdom sampling
        executive.set_eps();
        associative.set_eps();
        neuromodulation.set_eps();
        exteroceptive.set_eps();
        proprioceptive.set_eps();
        interoceptive.set_eps();
        
        //set constant for Rectified Adam
        double rho_inf = 2.0/(1.0-ADAM2)-1.0;
        double rho = rho_inf-2.0*(epoch+1)*pow(ADAM2,epoch+1)/(1.0-pow(ADAM2,epoch+1));
        double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/(rho_inf-4.0)/(rho_inf-2.0)/rho);
        double m_radam1 = 1.0/(1.0-pow(ADAM1,epoch+1));
        double l_radam2 = 1.0-pow(ADAM2,epoch+1);
        
        //multiple processing
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int s=0;s<SEQ_NUM;++s){
            int time_length = length[s];
            //Feedforward
            for(int t=0;t<time_length;++t){
                 executive.forward(t, s, epoch, "posterior");
                 associative.forward(executive.d, t, s, epoch, "posterior");
                 
                 neuromodulation.forward(executive.d, t, s, epoch, "posterior");
                 
                 exteroceptive.forward(associative.d, t, s, epoch, "posterior");
                 proprioceptive.forward(associative.d, t, s, epoch, "posterior");
                 interoceptive.forward(associative.d, t, s, epoch, "posterior");
                 
                 out_nm.forward(neuromodulation.d, t, s);
                 out_extero.forward(exteroceptive.d, t, s);
                 out_proprio.forward(proprioceptive.d, t, s);
                 out_intero.forward(interoceptive.d, t, s);
                 
                 for(int i = 0; i < x_num; ++i){
                     if(i<x_extero_num){
                         x_mean[s][t][i] = out_extero.output[s][t][i];
                         x_sigma_extero[s][t][i] = out_nm.output[s][t][i];
                     }else if(i<x_extero_num+x_proprio_num){
                         x_mean[s][t][i] = out_proprio.output[s][t][i-x_extero_num];
                         x_sigma_proprio[s][t][i-x_extero_num] = out_nm.output[s][t][i];
                     }else{
                         x_mean[s][t][i] = out_intero.output[s][t][i-x_extero_num-x_proprio_num];
                         x_sigma_intero[s][t][i-x_extero_num-x_proprio_num] = out_nm.output[s][t][i];
                     }
                 }
            }
            
            //Save generated sequence
            if((epoch) % SAVE_EPOCH == 0){
                executive.save_sequence_prediction(time_length, s, epoch);
                associative.save_sequence_prediction(time_length, s, epoch);
                neuromodulation.save_sequence_prediction(time_length, s, epoch);
                exteroceptive.save_sequence_prediction(time_length, s, epoch);
                proprioceptive.save_sequence_prediction(time_length, s, epoch);
                interoceptive.save_sequence_prediction(time_length, s, epoch);
                out_nm.save_sequence_prediction(time_length, s, epoch);
                out_extero.save_sequence_prediction(target_extero, time_length, s, epoch);
                out_proprio.save_sequence_prediction(target_proprio, time_length, s, epoch);
                out_intero.save_sequence_prediction(target_intero, time_length, s, epoch);
            }
            
            
            //Backward
            loss_seq[s] = 0.0;
            for(int t=time_length-1;t>=0;--t){
                out_intero.backward_nm(interoceptive.d, target_intero, x_sigma_intero, time_length, t, s);
                out_proprio.backward_nm(proprioceptive.d, target_proprio, x_sigma_proprio, time_length, t, s);
                out_extero.backward_nm(exteroceptive.d, target_extero, x_sigma_extero, time_length, t, s);
                out_nm.backward(neuromodulation.d, target_external, x_mean, time_length, t, s);
                interoceptive.backward(associative.d, out_intero.weight_ohd, out_intero.grad_internal_state_output, out_intero.tau, time_length, t, s);
                proprioceptive.backward(associative.d, out_proprio.weight_ohd, out_proprio.grad_internal_state_output, out_proprio.tau, time_length, t, s);
                
                exteroceptive.backward(associative.d, out_extero.weight_ohd, out_extero.grad_internal_state_output, out_extero.tau, time_length, t, s);
                
                neuromodulation.backward(executive.d, out_nm.weight_ohd, out_nm.grad_internal_state_output, out_nm.tau, time_length, t, s);
                
                //concatenate some variables in proprioceptive and exteroceptive areas to make backprop-inputs to the associative area
                for(int i=0;i<exteroceptive_d_num+proprioceptive_d_num+interoceptive_d_num;++i){
                    if(i<exteroceptive_d_num) {
                        exte_prop_intero_grad_internal_state_d[s][t][i] = exteroceptive.grad_internal_state_d[s][t][i];
                        exte_prop_intero_tau[i] = exteroceptive.tau[i];
                    }else if(i<exteroceptive_d_num+proprioceptive_d_num){
                        exte_prop_intero_grad_internal_state_d[s][t][i] = proprioceptive.grad_internal_state_d[s][t][i-exteroceptive_d_num];
                        exte_prop_intero_tau[i] = proprioceptive.tau[i-exteroceptive_d_num];
                    }else{
                        exte_prop_intero_grad_internal_state_d[s][t][i] = interoceptive.grad_internal_state_d[s][t][i-exteroceptive_d_num-proprioceptive_d_num];
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
                
                associative.backward(executive.d, exte_prop_intero_weight_ld, exte_prop_intero_grad_internal_state_d, exte_prop_intero_tau, time_length, t, s);
                
                //concatenate some variables in associative and neuromodulation areas to make backprop-inputs to the executive area
                for(int i=0;i<neuromodulation_d_num+associative_d_num;++i){
                    if(i<neuromodulation_d_num){
                        nm_ass_grad_internal_state_d[s][t][i] = neuromodulation.grad_internal_state_d[s][t][i];
                        nm_ass_tau[i] = neuromodulation.tau[i];
                    }else{
                        nm_ass_grad_internal_state_d[s][t][i] = associative.grad_internal_state_d[s][t][i-neuromodulation_d_num];
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
                
                executive.backward(nm_ass_weight_ld, nm_ass_grad_internal_state_d, nm_ass_tau, time_length, t, s);
                
                //loss
                for(int i=0; i<x_extero_num; ++i){
                    loss_seq[s] += out_extero.prediction_error[s][t][i];
                }
                for(int i=0; i<x_proprio_num; ++i){
                    loss_seq[s] += out_proprio.prediction_error[s][t][i];
                }
                for(int i=0; i<x_intero_num; ++i){
                    loss_seq[s] += out_intero.prediction_error[s][t][i];
                }
                for(int i=0; i<exteroceptive_z_num; ++i){
                    loss_seq[s] += exteroceptive.wkld[s][t][i];
                }
                for(int i=0; i<proprioceptive_z_num; ++i){
                    loss_seq[s] += proprioceptive.wkld[s][t][i];
                }
                for(int i=0; i<interoceptive_z_num; ++i){
                    loss_seq[s] += interoceptive.wkld[s][t][i];
                }
                for(int i=0; i<neuromodulation_z_num; ++i){
                    loss_seq[s] += neuromodulation.wkld[s][t][i];
                }
                for(int i=0; i<associative_z_num; ++i){
                    loss_seq[s] += associative.wkld[s][t][i];
                }
                for(int i=0; i<executive_z_num; ++i){
                    loss_seq[s] += executive.wkld[s][t][i];
                }
                
                //update adaptive vector (Rectified Adam) 
                interoceptive.update_parameter_radam_posterior(t, s, ALPHA, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                proprioceptive.update_parameter_radam_posterior(t, s, ALPHA, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                exteroceptive.update_parameter_radam_posterior(t, s, ALPHA, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                neuromodulation.update_parameter_radam_posterior(t, s, ALPHA, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                associative.update_parameter_radam_posterior(t, s, ALPHA, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                executive.update_parameter_radam_posterior(t, s, ALPHA, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                
            }
            
            //Save error
            if((epoch) % SAVE_EPOCH == 0){
                executive.save_sequence_error(time_length, s, epoch);
                associative.save_sequence_error(time_length, s, epoch);
                neuromodulation.save_sequence_error(time_length, s, epoch);
                exteroceptive.save_sequence_error(time_length, s, epoch);
                proprioceptive.save_sequence_error(time_length, s, epoch);
                interoceptive.save_sequence_error(time_length, s, epoch);
                out_extero.save_sequence_error(time_length, s, epoch);
                out_proprio.save_sequence_error(time_length, s, epoch);
                out_intero.save_sequence_error(time_length, s, epoch);
            }
            
        }
        
        loss = 0;
        
        //sum gradients over sequences
        for(int s=0;s<SEQ_NUM;++s){
            out_extero.sum_gradient(s);
            out_proprio.sum_gradient(s);
            out_intero.sum_gradient(s);
            out_nm.sum_gradient(s);
            exteroceptive.sum_gradient(s);
            proprioceptive.sum_gradient(s);
            interoceptive.sum_gradient(s);
            neuromodulation.sum_gradient(s);
            associative.sum_gradient(s);
            executive.sum_gradient(s);
            //compute free-energy accumulated over time steps and sequences
            loss += loss_seq[s]; 
        }
        
        //print error
        printf("%d: %f\n", epoch, loss);
        
        //Save weight
        if((epoch) % SAVE_EPOCH == 0){
            out_extero.save_parameter(epoch);
            out_proprio.save_parameter(epoch);
            out_intero.save_parameter(epoch);
            out_nm.save_parameter(epoch);
            exteroceptive.save_parameter(epoch);
            proprioceptive.save_parameter(epoch);
            interoceptive.save_parameter(epoch);
            neuromodulation.save_parameter(epoch);
            associative.save_parameter(epoch);
            executive.save_parameter(epoch);
        }
        
        //update synaptic weights
        out_extero.update_parameter_radam_weight(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        out_proprio.update_parameter_radam_weight(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        out_intero.update_parameter_radam_weight(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        out_nm.update_parameter_radam_weight(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        exteroceptive.update_parameter_radam_weight(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        proprioceptive.update_parameter_radam_weight(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        interoceptive.update_parameter_radam_weight(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        neuromodulation.update_parameter_radam_weight(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        associative.update_parameter_radam_weight(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        executive.update_parameter_radam_weight(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        //reset gradient
        executive.reset_gradient();
        associative.reset_gradient();
        neuromodulation.reset_gradient();
        exteroceptive.reset_gradient();
        proprioceptive.reset_gradient();
        interoceptive.reset_gradient();
        out_nm.reset_gradient();
        out_intero.reset_gradient();
        out_proprio.reset_gradient();
        out_extero.reset_gradient();
        
        
    }
    
    //Save trained result
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int s=0;s<SEQ_NUM;++s){
            int time_length = length[s];
            //Feedforward
            for(int t=0;t<time_length;++t){
                 executive.forward(t, s, EPOCH, "posterior");
                 associative.forward(executive.d, t, s, EPOCH, "posterior");
                 
                 neuromodulation.forward(executive.d, t, s, EPOCH, "posterior");
                 
                 exteroceptive.forward(associative.d, t, s, EPOCH, "posterior");
                 proprioceptive.forward(associative.d, t, s, EPOCH, "posterior");
                 interoceptive.forward(associative.d, t, s, EPOCH, "posterior");
                 
                 out_nm.forward(neuromodulation.d, t, s);
                 out_extero.forward(exteroceptive.d, t, s);
                 out_proprio.forward(proprioceptive.d, t, s);
                 out_intero.forward(interoceptive.d, t, s);
                 
                 for(int i = 0; i < x_num; ++i){
                     if(i<x_extero_num){
                         x_mean[s][t][i] = out_extero.output[s][t][i];
                         x_sigma_extero[s][t][i] = out_nm.output[s][t][i];
                     }else if(i<x_extero_num+x_proprio_num){
                         x_mean[s][t][i] = out_proprio.output[s][t][i-x_extero_num];
                         x_sigma_proprio[s][t][i-x_extero_num] = out_nm.output[s][t][i];
                     }else{
                         x_mean[s][t][i] = out_intero.output[s][t][i-x_extero_num-x_proprio_num];
                         x_sigma_intero[s][t][i-x_extero_num-x_proprio_num] = out_nm.output[s][t][i];
                     }
                 }
            }
            //Save generated sequence
            executive.save_sequence_prediction(time_length, s, EPOCH);
            associative.save_sequence_prediction(time_length, s, EPOCH);
            neuromodulation.save_sequence_prediction(time_length, s, EPOCH);
            exteroceptive.save_sequence_prediction(time_length, s, EPOCH);
            proprioceptive.save_sequence_prediction(time_length, s, EPOCH);
            interoceptive.save_sequence_prediction(time_length, s, EPOCH);
            out_nm.save_sequence_prediction(time_length, s, EPOCH);
            out_extero.save_sequence_prediction(target_extero, time_length, s, EPOCH);
            out_proprio.save_sequence_prediction(target_proprio, time_length, s, EPOCH);
            out_intero.save_sequence_prediction(target_intero, time_length, s, EPOCH);
            
            //Backward (compute prediction error and KLD)
            for(int t=time_length-1;t>=0;--t){
                out_intero.backward_nm(interoceptive.d, target_intero, x_sigma_intero, time_length, t, s);
                out_proprio.backward_nm(proprioceptive.d, target_proprio, x_sigma_proprio, time_length, t, s);
                out_extero.backward_nm(exteroceptive.d, target_extero, x_sigma_extero, time_length, t, s);
                out_nm.backward(neuromodulation.d, target_external, x_mean, time_length, t, s);
                interoceptive.backward(associative.d, out_intero.weight_ohd, out_intero.grad_internal_state_output, out_intero.tau, time_length, t, s);
                proprioceptive.backward(associative.d, out_proprio.weight_ohd, out_proprio.grad_internal_state_output, out_proprio.tau, time_length, t, s);
                exteroceptive.backward(associative.d, out_extero.weight_ohd, out_extero.grad_internal_state_output, out_extero.tau, time_length, t, s);
                
                neuromodulation.backward(executive.d, out_nm.weight_ohd, out_nm.grad_internal_state_output, out_nm.tau, time_length, t, s);
                
                //concatenate some variables in proprioceptive and exteroceptive areas to make backprop-inputs to the associative area
                for(int i=0;i<exteroceptive_d_num+proprioceptive_d_num+interoceptive_d_num;++i){
                    if(i<exteroceptive_d_num) {
                        exte_prop_intero_grad_internal_state_d[s][t][i] = exteroceptive.grad_internal_state_d[s][t][i];
                        exte_prop_intero_tau[i] = exteroceptive.tau[i];
                    }else if(i<exteroceptive_d_num+proprioceptive_d_num){
                        exte_prop_intero_grad_internal_state_d[s][t][i] = proprioceptive.grad_internal_state_d[s][t][i-exteroceptive_d_num];
                        exte_prop_intero_tau[i] = proprioceptive.tau[i-exteroceptive_d_num];
                    }else{
                        exte_prop_intero_grad_internal_state_d[s][t][i] = interoceptive.grad_internal_state_d[s][t][i-exteroceptive_d_num-proprioceptive_d_num];
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
                
                associative.backward(executive.d, exte_prop_intero_weight_ld, exte_prop_intero_grad_internal_state_d, exte_prop_intero_tau, time_length, t, s);
                
                //concatenate some variables in associative and neuromodulation areas to make backprop-inputs to the executive area
                for(int i=0;i<neuromodulation_d_num+associative_d_num;++i){
                    if(i<neuromodulation_d_num){
                        nm_ass_grad_internal_state_d[s][t][i] = neuromodulation.grad_internal_state_d[s][t][i];
                        nm_ass_tau[i] = neuromodulation.tau[i];
                    }else{
                        nm_ass_grad_internal_state_d[s][t][i] = associative.grad_internal_state_d[s][t][i-neuromodulation_d_num];
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
                
                executive.backward(nm_ass_weight_ld, nm_ass_grad_internal_state_d, nm_ass_tau, time_length, t, s);
            }
            //Save error
            executive.save_sequence_error(time_length, s, EPOCH);
            associative.save_sequence_error(time_length, s, EPOCH);
            neuromodulation.save_sequence_error(time_length, s, EPOCH);
            exteroceptive.save_sequence_error(time_length, s, EPOCH);
            proprioceptive.save_sequence_error(time_length, s, EPOCH);
            interoceptive.save_sequence_error(time_length, s, EPOCH);
            out_extero.save_sequence_error(time_length, s, EPOCH);
            out_proprio.save_sequence_error(time_length, s, EPOCH);
            out_intero.save_sequence_error(time_length, s, EPOCH);
    }
    out_extero.save_parameter(EPOCH);
    out_proprio.save_parameter(EPOCH);
    out_intero.save_parameter(EPOCH);
    out_nm.save_parameter(EPOCH);
    exteroceptive.save_parameter(EPOCH);
    proprioceptive.save_parameter(EPOCH);
    interoceptive.save_parameter(EPOCH);
    neuromodulation.save_parameter(EPOCH);
    associative.save_parameter(EPOCH);
    executive.save_parameter(EPOCH);
    
    gettimeofday(&end, NULL);
    float delta = end.tv_sec  - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
    printf("time %lf[s]\n", delta);
    
    return 0;
}
