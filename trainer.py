import os
import time
import segyio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from read_data import DATAIO
from utility import *

#GPU ARRANGMENT
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.set_visible_devices(gpus, 'GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4144)
            ])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

class Trainer(object):
    
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.version = config.version
        self.restore = config.restore

        self.inputs_path = config.inputs_path
        self.labels_path = config.labels_path
        self.tests_path  = config.tests_path
        self.preds_file  = config.preds_file

        self.epoch       = config.epoch
        self.valid_ratio = config.valid_ratio
        self.mute        = config.mute

        self.model_save_freq  = config.model_save_freq
        self.train_print_freq = config.train_print_freq
        self.valid_print_freq = config.valid_print_freq
        self.test_print_freq  = config.test_print_freq

        self.ver_dir   = config.ver_dir
        self.log_dir   = config.log_dir
        self.model_dir = config.model_dir
        self.test_dir  = config.test_dir
        self.img_dir   = config.img_dir

        self.preds_path = os.path.join(self.test_dir, self.preds_file)

        self.dataio = DATAIO(self.inputs_path, 
                             self.labels_path,
                             self.tests_path,
                             self.preds_path,
                             self.valid_ratio,
                            )

        self.name     = config.name
        self.ckpt_dir = config.ckpt_dir            
        self.m        = create_model(name=self.name)
        self.opt      = tf.optimizers.get(config.opt)
        self.opt.learning_rate = config.lr
        if isinstance(self.opt, tf.optimizers.Adam):
            self.opt.beta_1=config.beta1
            self.opt.beta_2=config.beta2

        self.SNR_min     = -200
        self.epoch_start = 0

    def valid_SNR(self, valids, labels):
        SNR = 10*tf.math.log(tf.math.reduce_sum(tf.math.square(labels)) / tf.math.reduce_sum(tf.math.square(valids - labels))) / tf.math.log(10.)
        return SNR

    def restore_models(self):
        if self.restore:
            latest = tf.train.latest_checkpoint(self.ckpt_dir)
            self.m.load_weights(latest).expect_partial()
            self.epoch_start = int(latest.split('/')[-1][3:].split('.')[0])

    def save_models(self, epoch):
        self.m.save_weights(os.path.join(self.ckpt_dir,   'cp-{:04d}.ckpt'.format(epoch)))

    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            preds     = self.m(inputs,  training=True)
            _loss     = tf.reduce_mean(tf.keras.losses.mse(labels, preds))
            gradients = tape.gradient(_loss, self.m.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.m.trainable_variables))
        return _loss


    def train(self):

        self.restore_models()

        log_path = os.path.join(self.log_dir, 'train_log_'+time.strftime('%Y%m%d_%H%M%S'))
        log_file = open(log_path, 'w')

        _prms, idxes_train, idxes_valid = self.dataio.read_data()

        total_loss_list   = []
        ave_loss_list     = []

        SNR_ave_list      = []
        SNR_total_list    = []
        SNR_label_list    = []

        for e in range(self.epoch_start, self.epoch_start + self.epoch):
            epoch = e + 1
            epoch_start_time = time.time()

            counter = 0
            loss_list = []
            print('\n\n'+'='*80, file=log_file)
            print(' '*25+'Start Training for ' + 'Epoch: {0:4d}'.format(epoch), file=log_file)
            for i in idxes_train:
                counter += 1
                inputs_batch, labels_batch = self.dataio.ret_train_gather(i, self.mute)
                inputs_batch = pre_process(inputs_batch, _prms)
                labels_batch = pre_process(labels_batch, _prms)

                _loss = self.train_step(inputs_batch, labels_batch)
                loss_list.append(_loss)

                if counter % self.train_print_freq == 0:
                    print("Epoch: {:04d}/{:04d}".format(epoch, self.epoch_start + self.epoch),' '*16,
                          "Counter: {:04d}".format(counter),' '*16,
                          "Loss: {:.6f}".format(_loss),
                           file=log_file)

            epoch_end_time = time.time()
            total_loss_list.append(loss_list)
            ave_loss = np.mean(loss_list)
            ave_loss_list.append(ave_loss)

            counter  = 0
            SNR_list = []
            print('='*80, file=log_file)
            print(' '*20+'Start Validation for ' + 'Epoch: {0:4d}'.format(epoch), file=log_file)
            for i in idxes_valid:
                counter += 1
                inputs_batch, labels_batch = self.dataio.ret_train_gather(i, self.mute)
                inputs_batch = pre_process(inputs_batch, _prms)
                labels_batch = pre_process(labels_batch, _prms)

                preds_batch   = self.m(inputs_batch, training=False)

                inputs_batch  = post_process(inputs_batch,  _prms)
                labels_batch  = post_process(labels_batch,  _prms)
                preds_batch   = post_process(preds_batch,   _prms)

                SNR           = self.valid_SNR(preds_batch, labels_batch)
                SNR_list.append(SNR)

                if epoch == 1:
                    SNR_label = self.valid_SNR(inputs_batch, labels_batch)
                    SNR_label_list.append(SNR_label)

                if counter % self.valid_print_freq == 0:
                    print('Epoch: {:04d}'.format(epoch),' '*15,
                          'Counter: {:04d}'.format(counter),' '*15,
                          'SNR: {:.6f}'.format(SNR),
                           file=log_file)


            ave_SNR       = np.mean(SNR_list)
            ave_SNR_label = np.mean(SNR_label_list)
            SNR_ave_list.append(ave_SNR)
            SNR_total_list.append(SNR_list)

            self.save_models(epoch)

            print('='*80, file=log_file)
            print(' '*20+'Epoch: {0:4d}'.format(epoch)+' '*5+'Average Indexes', file=log_file)
            print('Epoch: {:04d}'.format(epoch),' '*7,
                  'Loss: {:.6f}'.format(ave_loss),' '*7,
                  'SNR: {:.6f}'.format(ave_SNR),' '*7,
                  'Time: {:.4f}'.format((epoch_end_time - epoch_start_time) / 3600.0),
                   file=log_file
                 )
            print('='*80, file=log_file)


        img(self.img_dir, total_loss_list, ave_loss_list, SNR_ave_list, SNR_total_list, SNR_label_list)
        log_file.close()
        self.dataio.close_files()

        self.test()
        
    def test(self):
        self.restore_models()

        log_path = os.path.join(self.log_dir, 'test_log_'+time.strftime('%Y%m%d_%H%M%S'))
        log_file = open(log_path, 'w')

        _prms, idxes = self.dataio.read_inputs()
        inputs_file, inputs_spec = self.dataio.copy_()

        with segyio.create(self.preds_path, inputs_spec) as _preds_file:
            _preds_file.text[0] = inputs_file.text[0]
            _preds_file.bin     = inputs_file.bin
            _preds_file.header  = inputs_file.header

            test_start_time = time.time()

            counter  = 0
            print('\n'+'='*80, file=log_file)
            print(' '*30+'Start Testing for '+'Epoch: {0:4d}  '.format(self.epoch_start), file=log_file)
            for i in idxes:
                counter += 1
                inputs_batch = self.dataio.ret_test_gather(i, self.mute)
                inputs_batch = pre_process(inputs_batch, _prms)

                _preds_batch = self.m(inputs_batch, training=False)

                inputs_batch = post_process(inputs_batch,  _prms)
                _preds_batch = post_process(_preds_batch,  _prms)

                self.dataio.write_gather(i, _preds_batch, _preds_file, self.mute)

                if counter % self.test_print_freq == 0:
                    print('Counter: {:04d}'.format(counter),' '*20, 
                          'Done',
                           file=log_file)

            test_end_time = time.time()
            print('Predictions Saved!')

            print('='*80, file=log_file)
            print(' '*32+'Testing Indexes', file=log_file)
            print(' '*32+'Time: {:.4f} sec'.format((test_end_time - test_start_time)),
                   file=log_file)
            print('='*80, file=log_file)

            log_file.close()
            inputs_file.close()
            _preds_file.close()
