import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nets_lib import *

def create_model(name):
    model = eval(name)()
    model.build(input_shape = ())
    model.summary()
    return model

def pre_process(records, prms):
    after_norm = (records - prms[0]) / prms[1]
    return after_norm

def post_process(preds, prms):
    before_norm = preds * prms[1] + prms[0]
    return before_norm

def img(img_dir, m_total_loss_list, m_ave_loss_list, SNR_ave_list, SNR_total_list, SNR_label_list):
    
    loss_length  = len(m_total_loss_list[0])*len(m_total_loss_list)
    epoch_length = len(m_ave_loss_list)
    SNR_length   = len(SNR_total_list[0])*len(SNR_total_list)
    
    loss_x  = np.arange(1, loss_length+1)
    epoch_x = np.arange(1, epoch_length+1)
    SNR_x   = np.arange(1, SNR_length+1)
    
    m_loss_total = np.array(m_total_loss_list).reshape(loss_length)
    m_ave_loss   = np.array(m_ave_loss_list)
    SNR_total    = np.array(SNR_total_list).reshape(SNR_length)
    SNR_ave      = np.array(SNR_ave_list)
    
    m_loss_total.tofile(os.path.join(img_dir, 'Loss_total'))
    m_ave_loss.tofile(os.path.join(img_dir, 'Loss_ave'))
    SNR_total.tofile(os.path.join(img_dir, 'SNR_total'))
    SNR_ave.tofile(os.path.join(img_dir, 'SNR_ave'))
    
    plt.figure(figsize=(6,6))
    plt.plot(loss_x, m_loss_total, color="black", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Loss Value")
    plt.title("Loss Value in Iteration")
    plt.savefig(os.path.join(img_dir, 'Loss_total.png'))
    
    plt.figure(figsize=(6,6))
    plt.plot(SNR_x, SNR_total, color="black", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Validation SNR")
    plt.title("Validation SNR Value in Iteration")
    plt.savefig(os.path.join(img_dir, 'SNR_total.png'))
    
    plt.figure(figsize=(6,6))
    plt.plot(epoch_x, m_ave_loss, color="black", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss Value")
    plt.title("Average Loss Value in Epoch")
    plt.savefig(os.path.join(img_dir, 'Loss_ave.png'))
    
    plt.figure(figsize=(6,6))
    plt.plot(epoch_x, SNR_ave, color="black", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Validation SNR")
    plt.title("Average Validation SNR Value in Epoch")
    plt.savefig(os.path.join(img_dir, 'SNR_ave.png'))
