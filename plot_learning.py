#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pylab
import re
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from optparse import OptionParser

FOLDERNAME_GENERATE = "learning_generation"
FOLDERNAME_MODEL = "learning_model"
SEQUENCE_INDEX = "0"
SEQ_NUM = 1
EPOCH = "0100000"
T_MIN = 0
T_MAX = 5000
C_MAP = plt.cm.seismic
params = {"backend": "pdf",
          "font.family": "Arial",
          "axes.titlesize": 15,
          "axes.labelsize": 15,
          "font.size": 15,
          "legend.fontsize":15,
          "xtick.labelsize": 15,
          "ytick.labelsize": 15,
          "text.usetex": False,
          "savefig.facecolor": "1.0"}
pylab.rcParams.update(params)

def read_parameter(f):
    r = {}
    r["c_state_size"] = re.compile(r"^# c_state_size")
    r["out_state_size"] = re.compile(r"^# out_state_size")
    r_comment = re.compile(r'^#')
    params = {}
    for line in f:
        for k,v in r.iteritems():
            if (v.match(line)):
                x = line.split('=')[1]
                if k == 'target':
                    m = int(v.match(line).group(1))
                    if (k in params):
                        params[k][m] = x
                    else:
                        params[k] = {m:x}
                else:
                    params[k] = x

        if (r_comment.match(line) == None):
            break
    f.seek(0)
    return params
    
class plot_rnn(object):
    def __init__(self,
                 filename, filename_executive_in_p_mu, filename_executive_in_p_sigma, filename_executive_a_mu, filename_executive_a_sigma, filename_executive_d, filename_associative_in_p_mu, filename_associative_in_p_sigma, filename_associative_a_mu, filename_associative_a_sigma, filename_associative_d, filename_neuromodulation_in_p_mu, filename_neuromodulation_in_p_sigma, filename_neuromodulation_a_mu, filename_neuromodulation_a_sigma, filename_neuromodulation_d, filename_output_nm, filename_proprioceptive_in_p_mu, filename_proprioceptive_in_p_sigma, filename_proprioceptive_a_mu, filename_proprioceptive_a_sigma, filename_proprioceptive_d, filename_output_proprio, filename_target_proprio, filename_exteroceptive_in_p_mu, filename_exteroceptive_in_p_sigma, filename_exteroceptive_a_mu, filename_exteroceptive_a_sigma, filename_exteroceptive_d, filename_output_extero, filename_target_extero, filename_interoceptive_in_p_mu, filename_interoceptive_in_p_sigma, filename_interoceptive_a_mu, filename_interoceptive_a_sigma, filename_interoceptive_d, filename_output_intero, filename_target_intero):
        self.figure_name = filename
        
        #executive area
        self.state_filename_executive_in_p_mu = filename_executive_in_p_mu
        self.state_filename_executive_in_p_sigma = filename_executive_in_p_sigma
        self.state_filename_executive_a_mu = filename_executive_a_mu
        self.state_filename_executive_a_sigma = filename_executive_a_sigma
        self.state_filename_executive_d = filename_executive_d
        
        #associative area
        self.state_filename_associative_in_p_mu = filename_associative_in_p_mu
        self.state_filename_associative_in_p_sigma = filename_associative_in_p_sigma
        self.state_filename_associative_a_mu = filename_associative_a_mu
        self.state_filename_associative_a_sigma = filename_associative_a_sigma
        self.state_filename_associative_d = filename_associative_d
        
        #neuromodulation area
        self.state_filename_neuromodulation_in_p_mu = filename_neuromodulation_in_p_mu
        self.state_filename_neuromodulation_in_p_sigma = filename_neuromodulation_in_p_sigma
        self.state_filename_neuromodulation_a_mu = filename_neuromodulation_a_mu
        self.state_filename_neuromodulation_a_sigma = filename_neuromodulation_a_sigma
        self.state_filename_neuromodulation_d = filename_neuromodulation_d
        
        
        #exteroceptive area
        self.state_filename_exteroceptive_in_p_mu = filename_exteroceptive_in_p_mu
        self.state_filename_exteroceptive_in_p_sigma = filename_exteroceptive_in_p_sigma
        self.state_filename_exteroceptive_a_mu = filename_exteroceptive_a_mu
        self.state_filename_exteroceptive_a_sigma = filename_exteroceptive_a_sigma
        self.state_filename_exteroceptive_d = filename_exteroceptive_d
        
        
        #interoceptive area
        self.state_filename_interoceptive_in_p_mu = filename_interoceptive_in_p_mu
        self.state_filename_interoceptive_in_p_sigma = filename_interoceptive_in_p_sigma
        self.state_filename_interoceptive_a_mu = filename_interoceptive_a_mu
        self.state_filename_interoceptive_a_sigma = filename_interoceptive_a_sigma
        self.state_filename_interoceptive_d = filename_interoceptive_d
        
        #proprioceptive area
        self.state_filename_proprioceptive_in_p_mu = filename_proprioceptive_in_p_mu
        self.state_filename_proprioceptive_in_p_sigma = filename_proprioceptive_in_p_sigma
        self.state_filename_proprioceptive_a_mu = filename_proprioceptive_a_mu
        self.state_filename_proprioceptive_a_sigma = filename_proprioceptive_a_sigma
        self.state_filename_proprioceptive_d = filename_proprioceptive_d
        
        #output
        self.state_filename_output_extero = filename_output_extero
        self.state_filename_output_intero = filename_output_intero
        self.state_filename_output_proprio = filename_output_proprio
        self.state_filename_output_nm = filename_output_nm
        
        #target
        self.state_filename_target_extero = filename_target_extero
        self.state_filename_target_intero = filename_target_intero
        self.state_filename_target_proprio = filename_target_proprio

    def add_info(self, ax, title, xlim, ylim, xlabel, ylabel):
        if title != None:
            ax.set_title(title)

        if xlim != None:
            ax.set_xlim(xlim)
        else:
            ax.set_xticks([])

        if xlabel != None:
            ax.set_xlabel(xlabel)

        if ylim != None:
            ax.set_ylim(ylim)
            ax.set_yticks((ylim[0], (ylim[0] + ylim[1]) / 2.0, ylim[1]))
        if ylabel != None:
            ax.set_ylabel(ylabel)

        ax.grid(True) 
        
    def set_no_yticks(self, ax):
        ax.set_yticks([])

    def configure(self, fig_matrix, width, height):
        fig = plt.figure(figsize = (1.5*width * fig_matrix[1], height * fig_matrix[0]))
        gs = gridspec.GridSpec(fig_matrix[0], fig_matrix[1])
        axes = [fig.add_subplot(gs[i, j]) for i in range(fig_matrix[0]) for j in range(fig_matrix[1])]
        return fig, gs, axes

    def plot_colormap(self, ax, state, range):
        im = ax.imshow(state.T, vmin = range[0], vmax = range[1], aspect = "auto", interpolation = "nearest", cmap = C_MAP)
        if state.shape[0] != 1:
            ax.set_xlim(0, state.shape[0])
            ax.set_ylim(-0.5, state.shape[1] - 0.5)
            ax.set_yticks((0, state.shape[1] -1))
        return im

    def state(self, tmin, tmax, width, height):
        fig_matrix = [27, 1]
        fig, gs, axes = self.configure(fig_matrix, width, height)
        
        #executive area
        executive_in_p_mu = np.loadtxt(self.state_filename_executive_in_p_mu)
        executive_in_p_sigma = np.loadtxt(self.state_filename_executive_in_p_sigma)
        executive_a_mu = np.loadtxt(self.state_filename_executive_a_mu)
        executive_a_sigma = np.loadtxt(self.state_filename_executive_a_sigma)
        executive_d = np.loadtxt(self.state_filename_executive_d)
        
        #associative area
        associative_in_p_mu = np.loadtxt(self.state_filename_associative_in_p_mu)
        associative_in_p_sigma = np.loadtxt(self.state_filename_associative_in_p_sigma)
        associative_a_mu = np.loadtxt(self.state_filename_associative_a_mu)
        associative_a_sigma = np.loadtxt(self.state_filename_associative_a_sigma)
        associative_d = np.loadtxt(self.state_filename_associative_d)
        
        #neuromodulation area
        neuromodulation_in_p_mu = np.loadtxt(self.state_filename_neuromodulation_in_p_mu)
        neuromodulation_in_p_sigma = np.loadtxt(self.state_filename_neuromodulation_in_p_sigma)
        neuromodulation_a_mu = np.loadtxt(self.state_filename_neuromodulation_a_mu)
        neuromodulation_a_sigma = np.loadtxt(self.state_filename_neuromodulation_a_sigma)
        neuromodulation_d = np.loadtxt(self.state_filename_neuromodulation_d)
        
        
        #exteroceptive area
        exteroceptive_in_p_mu = np.loadtxt(self.state_filename_exteroceptive_in_p_mu)
        exteroceptive_in_p_sigma = np.loadtxt(self.state_filename_exteroceptive_in_p_sigma)
        exteroceptive_a_mu = np.loadtxt(self.state_filename_exteroceptive_a_mu)
        exteroceptive_a_sigma = np.loadtxt(self.state_filename_exteroceptive_a_sigma)
        exteroceptive_d = np.loadtxt(self.state_filename_exteroceptive_d)
        
        #interoceptive area
        interoceptive_in_p_mu = np.loadtxt(self.state_filename_interoceptive_in_p_mu)
        interoceptive_in_p_sigma = np.loadtxt(self.state_filename_interoceptive_in_p_sigma)
        interoceptive_a_mu = np.loadtxt(self.state_filename_interoceptive_a_mu)
        interoceptive_a_sigma = np.loadtxt(self.state_filename_interoceptive_a_sigma)
        interoceptive_d = np.loadtxt(self.state_filename_interoceptive_d)
        
        #proprioceptive area
        proprioceptive_in_p_mu = np.loadtxt(self.state_filename_proprioceptive_in_p_mu)
        proprioceptive_in_p_sigma = np.loadtxt(self.state_filename_proprioceptive_in_p_sigma)
        proprioceptive_a_mu = np.loadtxt(self.state_filename_proprioceptive_a_mu)
        proprioceptive_a_sigma = np.loadtxt(self.state_filename_proprioceptive_a_sigma)
        proprioceptive_d = np.loadtxt(self.state_filename_proprioceptive_d)
        
        #output
        extero_size = 2
        proprio_size = 2
        intero_size = 1
        output_extero = np.loadtxt(self.state_filename_output_extero)
        output_proprio = np.loadtxt(self.state_filename_output_proprio)
        output_intero = np.loadtxt(self.state_filename_output_intero)
        output_nm = np.loadtxt(self.state_filename_output_nm)
        sigma_extero = output_nm[:, :extero_size]
        sigma_proprio = output_nm[:, extero_size:extero_size+proprio_size]
        sigma_intero = output_nm[:, extero_size+proprio_size:]
        
        #target
        target_extero = np.loadtxt(self.state_filename_target_extero)
        target_proprio = np.loadtxt(self.state_filename_target_proprio)
        target_intero = np.loadtxt(self.state_filename_target_intero)
        
        cmap = plt.get_cmap('tab20')
        
        #executive 
        for x in range(executive_a_mu.shape[1]):
            axes[0].plot(np.tanh(executive_a_mu[:, x]), linestyle="solid", linewidth="1", color=cmap(x))
            axes[0].plot(np.tanh(executive_in_p_mu[:, x]), linestyle="dashed", linewidth="1", color=cmap(x))
            axes[1].plot(np.exp(executive_a_sigma[:, x]), linestyle="solid", linewidth="1", color=cmap(x))
            axes[1].plot(np.exp(executive_in_p_sigma[:, x]), linestyle="dashed", linewidth="1", color=cmap(x))
        self.add_info(axes[0], None, None, (-1.2,1.2), None, "Exe. mu")
        self.add_info(axes[1], None, None, (0.0, 1.5), None, "Exe. sigma")
        axes[2].plot(np.tanh(executive_d), linestyle="solid", linewidth="1")
        self.add_info(axes[2], None, None, (-1.2,1.2), None, "Exe. d")
        
        #associative
        for x in range(associative_a_mu.shape[1]):
            axes[3].plot(np.tanh(associative_a_mu[:, x]), linestyle="solid", linewidth="1", color=cmap(x))
            axes[3].plot(np.tanh(associative_in_p_mu[:, x]), linestyle="dashed", linewidth="1", color=cmap(x))
            
            axes[4].plot(np.exp(associative_a_sigma[:, x]), linestyle="solid", linewidth="1", color=cmap(x))
            axes[4].plot(np.exp(associative_in_p_sigma[:, x]), linestyle="dashed", linewidth="1", color=cmap(x))
        self.add_info(axes[3], None, None, (-1.2,1.2), None, "Ass. mu")
        self.add_info(axes[4], None, None, (0.0, 1.5), None, "Ass. sigma")
        axes[5].plot(np.tanh(associative_d), linestyle="solid", linewidth="1")
        self.add_info(axes[5], None, None, (-1.2,1.2), None, "Ass. d")
        
        #neuromodulation
        axes[6].plot(np.tanh(neuromodulation_a_mu), linestyle="solid", linewidth="1", color=cmap(0))
        axes[6].plot(np.tanh(neuromodulation_in_p_mu), linestyle="dashed", linewidth="1", color=cmap(0))
        self.add_info(axes[6], None, None, (-1.2,1.2), None, "NM. mu")
        axes[7].plot(np.exp(neuromodulation_a_sigma), linestyle="solid", linewidth="1", color=cmap(0))
        axes[7].plot(np.exp(neuromodulation_in_p_sigma), linestyle="dashed", linewidth="1", color=cmap(0))
        self.add_info(axes[7], None, None, (0.0, 3.0), None, "NM. sigma")
        axes[8].plot(np.tanh(neuromodulation_d), linestyle="solid", linewidth="1")
        self.add_info(axes[8], None, None, (-1.2,1.2), None, "NM. d")
        
        #exteroceptive area 
        for x in range(exteroceptive_a_mu.shape[1]):
            axes[9].plot(np.tanh(exteroceptive_a_mu[:, x]), linestyle="solid", linewidth="1", color=cmap(x))
            axes[9].plot(np.tanh(exteroceptive_in_p_mu[:, x]), linestyle="dashed", linewidth="1", color=cmap(x))
            axes[10].plot(np.exp(exteroceptive_a_sigma[:, x]), linestyle="solid", linewidth="1", color=cmap(x))
            axes[10].plot(np.exp(exteroceptive_in_p_sigma[:, x]), linestyle="dashed", linewidth="1", color=cmap(x))
        self.add_info(axes[9], None, None, (-1.2,1.2), None, "Extero. mu")
        self.add_info(axes[10], None, None, (0.0, 3.0), None, "Extero. sigma")
        axes[11].plot(np.tanh(exteroceptive_d), linestyle="solid", linewidth="1")
        self.add_info(axes[11], None, None, (-1.2,1.2), None, "Extero. d")
        
        #proprioceptive area
        for x in range(proprioceptive_a_mu.shape[1]):
            axes[12].plot(np.tanh(proprioceptive_a_mu[:, x]), linestyle="solid", linewidth="1", color=cmap(x))
            axes[12].plot(np.tanh(proprioceptive_in_p_mu[:, x]), linestyle="dashed", linewidth="1", color=cmap(x))
            axes[13].plot(np.exp(proprioceptive_a_sigma[:, x]), linestyle="solid", linewidth="1", color=cmap(x))
            axes[13].plot(np.exp(proprioceptive_in_p_sigma[:, x]), linestyle="dashed", linewidth="1", color=cmap(x))
        self.add_info(axes[12], None, None, (-1.2,1.2), None, "Prop. mu")
        self.add_info(axes[13], None, None, (0.0, 1.5), None, "Prop. sigma")
        axes[14].plot(np.tanh(proprioceptive_d), linestyle="solid", linewidth="1")
        self.add_info(axes[14], None, None, (-1.2,1.2), None, "Prop. d")
        
        #interoceptive area
        #for x in range(1):
        axes[15].plot(np.tanh(interoceptive_a_mu), linestyle="solid", linewidth="1", color=cmap(0))
        axes[15].plot(np.tanh(interoceptive_in_p_mu), linestyle="dashed", linewidth="1", color=cmap(0))
        self.add_info(axes[15], None, None, (-1.2,1.2), None, "Intero. mu")
        axes[16].plot(np.exp(interoceptive_a_sigma), linestyle="solid", linewidth="1", color=cmap(0))
        axes[16].plot(np.exp(interoceptive_in_p_sigma), linestyle="dashed", linewidth="1", color=cmap(0))
        self.add_info(axes[16], None, None, (0.0, 1.5), None, "Intero. sigma")
        axes[17].plot(np.tanh(interoceptive_d), linestyle="solid", linewidth="1")
        self.add_info(axes[17], None, None, (-1.2,1.2), None, "Intero. d")
                
        #output
        axes[18].plot(output_extero, linestyle="solid", linewidth="1")
        self.add_info(axes[18], None, None, (-1.2,1.2), None, "ExOut pred.")
        axes[19].plot(sigma_extero, linestyle="solid", linewidth="1") #extero
        self.add_info(axes[19], None, None, (0.0, 0.5), None, "ExOut sigma")
        axes[20].plot(target_extero, linestyle="solid", linewidth="1")
        self.add_info(axes[20], None, None, (-1.2,1.2), None, "ExOut target")
        
        axes[21].plot(output_proprio, linestyle="solid", linewidth="1")
        self.add_info(axes[21], None, None, (-1.2,1.2), None, "PropOut pred.")
        axes[22].plot(sigma_proprio, linestyle="solid", linewidth="1") #extero
        self.add_info(axes[22], None, None, (0.0, 0.5), None, "PropOut sigma")
        axes[23].plot(target_proprio, linestyle="solid", linewidth="1")
        self.add_info(axes[23], None, None, (-1.2,1.2), None, "PropOut target")
        
        axes[24].plot(output_intero, linestyle="solid", linewidth="1")
        self.add_info(axes[24], None, None, (-1.2,1.2), None, "InteOut pred.")
        axes[25].plot(sigma_intero, linestyle="solid", linewidth="1") #extero
        self.add_info(axes[25], None, None, (0.0, 0.5), None, "InteOut sigma")
        axes[26].plot(target_intero, linestyle="solid", linewidth="1")
        self.add_info(axes[26], None, None, (-1.2,1.2), "Time", "InteOut target") #(0, 200)
        
        for ax in axes:
            ax.set_xlim(tmin, tmax)
            
        fig.savefig(self.figure_name, format="pdf",dpi=600)
        #fig.show()
        
def main():
    parser = OptionParser()
    parser.add_option("-s", "--sequence", dest="sequence",
                      help="sequence index", metavar=SEQUENCE_INDEX, default=SEQUENCE_INDEX)
    parser.add_option("-e", "--epoch", dest="epoch",
                      help="epoch", metavar="EPOCH", default= EPOCH)
    (options, args) = parser.parse_args()
    
    for s in range(SEQ_NUM):
        filename = "./figure/generate_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".pdf"
        #executive area
        filename_executive_in_p_mu = "./" + FOLDERNAME_GENERATE + "/executive" + "/in_p_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_executive_in_p_sigma = "./" + FOLDERNAME_GENERATE + "/executive" + "/in_p_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_executive_a_mu = "./" + FOLDERNAME_MODEL + "/executive" + "/a_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_executive_a_sigma = "./" + FOLDERNAME_MODEL + "/executive" + "/a_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_executive_d = "./" + FOLDERNAME_GENERATE + "/executive" + "/d_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"  
          
        #associative area
        filename_associative_in_p_mu = "./" + FOLDERNAME_GENERATE + "/associative" + "/in_p_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_associative_in_p_sigma = "./" + FOLDERNAME_GENERATE + "/associative" + "/in_p_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_associative_a_mu = "./" + FOLDERNAME_MODEL + "/associative" + "/a_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_associative_a_sigma = "./" + FOLDERNAME_MODEL + "/associative" + "/a_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_associative_d = "./" + FOLDERNAME_GENERATE + "/associative" + "/d_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        
        #neuromodulation area
        filename_neuromodulation_in_p_mu = "./" + FOLDERNAME_GENERATE + "/neuromodulation" + "/in_p_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_neuromodulation_in_p_sigma = "./" + FOLDERNAME_GENERATE + "/neuromodulation" + "/in_p_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_neuromodulation_a_mu = "./" + FOLDERNAME_MODEL + "/neuromodulation" + "/a_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_neuromodulation_a_sigma = "./" + FOLDERNAME_MODEL + "/neuromodulation" + "/a_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_neuromodulation_d = "./" + FOLDERNAME_GENERATE + "/neuromodulation" + "/d_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        
        #exteroceptive area 
        filename_exteroceptive_in_p_mu = "./" + FOLDERNAME_GENERATE + "/exteroceptive" + "/in_p_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_exteroceptive_in_p_sigma = "./" + FOLDERNAME_GENERATE + "/exteroceptive" + "/in_p_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_exteroceptive_a_mu = "./" + FOLDERNAME_MODEL + "/exteroceptive" + "/a_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_exteroceptive_a_sigma = "./" + FOLDERNAME_MODEL + "/exteroceptive" + "/a_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_exteroceptive_d = "./" + FOLDERNAME_GENERATE + "/exteroceptive" + "/d_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        
        #proprioceptive area
        filename_proprioceptive_in_p_mu = "./" + FOLDERNAME_GENERATE + "/proprioceptive" + "/in_p_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_proprioceptive_in_p_sigma = "./" + FOLDERNAME_GENERATE + "/proprioceptive" + "/in_p_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_proprioceptive_a_mu = "./" + FOLDERNAME_MODEL + "/proprioceptive" + "/a_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_proprioceptive_a_sigma = "./" + FOLDERNAME_MODEL + "/proprioceptive" + "/a_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_proprioceptive_d = "./" + FOLDERNAME_GENERATE + "/proprioceptive" + "/d_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        
        #interoceptive area
        filename_interoceptive_in_p_mu = "./" + FOLDERNAME_GENERATE + "/interoceptive" + "/in_p_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_interoceptive_in_p_sigma = "./" + FOLDERNAME_GENERATE + "/interoceptive" + "/in_p_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_interoceptive_a_mu = "./" + FOLDERNAME_MODEL + "/interoceptive" + "/a_mu_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_interoceptive_a_sigma = "./" + FOLDERNAME_MODEL + "/interoceptive" + "/a_sigma_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_interoceptive_d = "./" + FOLDERNAME_GENERATE + "/interoceptive" + "/d_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        
        #output
        filename_output_proprio = "./" + FOLDERNAME_GENERATE + "/out_proprio" + "/output_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_output_extero = "./" + FOLDERNAME_GENERATE + "/out_extero" + "/output_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_output_intero = "./" + FOLDERNAME_GENERATE + "/out_intero" + "/output_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_output_nm = "./" + FOLDERNAME_GENERATE + "/out_nm" + "/output_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        
        #target
        filename_target_proprio = "./" + FOLDERNAME_GENERATE + "/out_proprio" + "/target_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_target_extero = "./" + FOLDERNAME_GENERATE + "/out_extero" + "/target_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        filename_target_intero = "./" + FOLDERNAME_GENERATE + "/out_intero" + "/target_{:0>7}".format(s) + "_{:0>7}".format(options.epoch) + ".txt"
        #filename_target = "./target" + "/target_{:0>7}".format(s) + ".txt"
        
        
        plot = plot_rnn(filename, filename_executive_in_p_mu, filename_executive_in_p_sigma, filename_executive_a_mu, filename_executive_a_sigma, filename_executive_d, filename_associative_in_p_mu, filename_associative_in_p_sigma, filename_associative_a_mu, filename_associative_a_sigma, filename_associative_d, filename_neuromodulation_in_p_mu, filename_neuromodulation_in_p_sigma, filename_neuromodulation_a_mu, filename_neuromodulation_a_sigma, filename_neuromodulation_d, filename_output_nm, filename_proprioceptive_in_p_mu, filename_proprioceptive_in_p_sigma, filename_proprioceptive_a_mu, filename_proprioceptive_a_sigma, filename_proprioceptive_d, filename_output_proprio, filename_target_proprio, filename_exteroceptive_in_p_mu, filename_exteroceptive_in_p_sigma, filename_exteroceptive_a_mu, filename_exteroceptive_a_sigma, filename_exteroceptive_d, filename_output_extero, filename_target_extero, filename_interoceptive_in_p_mu, filename_interoceptive_in_p_sigma, filename_interoceptive_a_mu, filename_interoceptive_a_sigma, filename_interoceptive_d, filename_output_intero, filename_target_intero)
        plot.state(T_MIN, T_MAX, 27, 2)


if __name__ == "__main__":
    main()
