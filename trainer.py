import os
import time
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
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5144)
            ])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

class Trainer(object):
    
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.version = config.version
        self.gan     = config.gan
        self.restore = config.restore

        self.inputs_path = config.inputs_path
        self.labels_path = config.labels_path
        self.preds_file  = config.preds_file

        self.epoch_start = config.epoch_start
        self.epoch       = config.epoch
        self.batch_size  = config.batch_size
        self.valid_ratio = config.valid_ratio
        self.mute        = config.mute

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
                               self.preds_path,
                               self.batch_size,
                               self.valid_ratio,
                               )

        if self.gan:
            self.d_name     = config.d_name
            self.g_name     = config.g_name
            self.d_ckpt_dir = config.d_ckpt_dir
            self.g_ckpt_dir = config.g_ckpt_dir

            # create models
            self.d = create_model(name=self.d_name)
            self.g = create_model(name=self.g_name)

            # initial optimizers
            self.d_opt = tf.optimizers.get(config.d_opt)
            self.d_opt.learning_rate = config.d_lr
            if isinstance(self.d_opt, tf.optimizers.Adam):
                self.d_opt.beta_1=config.beta1
                self.d_opt.beta_2=config.beta2
            self.g_opt = tf.optimizers.get(config.g_opt)
            self.g_opt.learning_rate = config.g_lr
            if isinstance(self.g_opt, tf.optimizers.Adam):
                self.g_opt.beta_1=config.beta1
                self.g_opt.beta_2=config.beta2

        else:
            self.name     = config.name
            self.ckpt_dir = config.ckpt_dir            
            self.m        = create_model(name=self.name)
            self.opt      = tf.optimizers.get(config.opt)
            self.opt.learning_rate = config.lr
            if isinstance(self.opt, tf.optimizers.Adam):
                self.opt.beta_1=config.beta1
                self.opt.beta_2=config.beta2

        self.SNR_min = -200

    def valid_SNR(self, valids, labels):
        SNR = 10*tf.math.log(tf.math.reduce_sum(tf.math.square(labels)) / tf.math.reduce_sum(tf.math.square(valids - labels))) / tf.math.log(10.)
        return SNR

    def test_SNR(self, preds, labels):
        SNR = 10*tf.math.log(tf.math.reduce_sum(tf.math.square(preds)) / tf.math.reduce_sum(tf.math.square(preds - labels))) / tf.math.log(10.)
        return SNR

    def restore_models(self):
        if self.restore:
            if self.gan:
                d_latest = tf.train.latest_checkpoint(self.d_ckpt_dir)
                self.d.load_weights(d_latest)
                g_latest = tf.train.latest_checkpoint(self.g_ckpt_dir)
                self.g.load_weights(g_latest)
                self.epoch_start = int(g_latest.split('/')[-1][3:].split('.')[0])
            else:
                latest = tf.train.latest_checkpoint(self.ckpt_dir)
                self.m.load_weights(latest)
                self.epoch_start = int(latest.split('/')[-1][3:].split('.')[0])

    def save_models(self, epoch):
        if epoch - self.epoch_start == 1:
            self.SNR_min = SNR
            if self.gan:
                self.d.save_weights(os.path.join(self.d_ckpt_dir, 'cp-{:04d}.ckpt'.format(epoch)))
                self.g.save_weights(os.path.join(self.g_ckpt_dir, 'cp-{:04d}.ckpt'.format(epoch)))
            else:
                self.m.save_weights(os.path.join(self.ckpt_dir,   'cp-{:04d}.ckpt'.format(epoch)))
        else:
            if SNR >= self.SNR_min:
                self.SNR_min = SNR
                if self.gan:
                    self.d.save_weights(os.path.join(self.d_ckpt_dir, 'cp-{:04d}.ckpt'.format(epoch)))
                    self.g.save_weights(os.path.join(self.g_ckpt_dir, 'cp-{:04d}.ckpt'.format(epoch)))
                else:
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

        inputs, labels, cdpt_max_list, ns, _prms, idxes_train, idxes_valid, idxes = self.dataio.read_data()

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
                trace_start  = sum(cdpt_max_list[:i])
                trace_end    = trace_start + cdpt_max_list[i]

                inputs_batch = inputs[trace_start:trace_end, self.mute:ns]
                inputs_batch = inputs_batch.reshape(-1, cdpt_max_list[i], ns-self.mute, 1)
                labels_batch = labels[trace_start:trace_end, self.mute:ns]
                labels_batch = labels_batch.reshape(-1, cdpt_max_list[i], ns-self.mute, 1)

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
                trace_start  = sum(cdpt_max_list[:i])
                trace_end    = trace_start + cdpt_max_list[i]

                inputs_batch = inputs[trace_start:trace_end, self.mute:ns]
                inputs_batch = inputs_batch.reshape(-1, cdpt_max_list[i], ns-self.mute, 1)
                labels_batch = labels[trace_start:trace_end, self.mute:ns]
                labels_batch = labels_batch.reshape(-1, cdpt_max_list[i], ns-self.mute, 1)

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

            self.save_models(epoch, ave_SNR)

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

        self.test()

    def gradient_penalty(self, inputs, g_preds):
        t = tf.random.uniform([self.batch_size,1,1,1])
        t = tf.broadcast_to(t, inputs.shape)
        interp = t * inputs + (1 - t) * g_preds
        with tf.GradientTape() as tape:
            tape.watch([interp])
            d_interp_logits = self.d(interp, training=True)
        grads = tape.gradient(d_interp_logits, interp)
        grads = tf.reshape(grads, [grads.shape[0],-1])
        gp = tf.norm(grads,axis=1)
        gp = tf.reduce_mean((gp-1.)**2)

        return gp

    def train_d_step(self, inputs, labels):
        with tf.GradientTape() as tape_d:
            g_preds     = self.g(inputs,  training=False)
            d_real_pred = self.d(labels,  training=True)
            d_fake_pred = self.d(g_preds, training=True)

            d_real_loss = tf.reduce_mean(d_real_pred)
            d_fake_loss = tf.reduce_mean(d_fake_pred)

            lamda_d = 10.
            gp = self.gradient_penalty(inputs, g_preds)

            d_loss = - d_real_loss + d_fake_loss + lamda_d * gp

        gradients = tape_d.gradient(d_loss, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients, self.d.trainable_variables))
        
        return d_loss

    def train_g_step(self, inputs, labels):
        with tf.GradientTape() as tape_g:
            g_preds     = self.g(inputs,  training=True)
            d_fake_pred = self.d(g_preds, training=False)
            d_fake_loss = tf.reduce_mean(d_fake_pred)

            lamda_g      = 1000
            regular_loss = tf.reduce_mean(tf.keras.losses.mse(labels, g_preds))

            g_loss = - d_fake_loss + lamda_g * regular_loss

        gradients = tape_g.gradient(g_loss, self.g.trainable_variables)
        self.g_opt.apply_gradients(zip(gradients, self.g.trainable_variables))
        
        return g_loss

    def train_gan(self):

        self.restore_models()

        log_path = os.path.join(self.log_dir, 'train_log_'+time.strftime('%Y%m%d_%H%M%S'))
        log_file = open(log_path, 'w')

        inputs, labels, cdpt_max_list, ns, _prms, idxes_train, idxes_valid, idxes = self.dataio.read_data()

        d_total_loss_list = []
        g_total_loss_list = []
        d_ave_loss_list   = []
        g_ave_loss_list   = []

        SNR_ave_list      = []
        SNR_total_list    = []
        SNR_label_list    = []

        for e in range(self.epoch_start, self.epoch_start + self.epoch):
            epoch = e + 1
            epoch_start_time = time.time()

            counter = 0
            counter = 0
            d_loss_list = []
            g_loss_list = []
            print('\n\n'+'='*80, file=log_file)
            print(' '*25+'Start Training for ' + 'Epoch: {0:4d}'.format(epoch), file=log_file)
            for i in idxes_train:
                counter += 1
                trace_start  = sum(cdpt_max_list[:i])
                trace_end    = trace_start + cdpt_max_list[i]

                inputs_batch = inputs[trace_start:trace_end, self.mute:ns]
                inputs_batch = inputs_batch.reshape(-1, cdpt_max_list[i], ns-self.mute, 1)
                labels_batch = labels[trace_start:trace_end, self.mute:ns]
                labels_batch = labels_batch.reshape(-1, cdpt_max_list[i], ns-self.mute, 1)

                inputs_batch = pre_process(inputs_batch, _prms)
                labels_batch = pre_process(labels_batch, _prms)

                d_loss = self.train_d_step(inputs_batch, labels_batch)
                d_loss_list.append(d_loss)
                g_loss = self.train_g_step(inputs_batch, labels_batch)
                g_loss_list.append(g_loss)

                if counter % self.train_print_freq == 0:
                    print("Epoch: {:04d}/{:04d}".format(epoch, self.epoch_start + self.epoch),' '*4,
                          "Counter: {:04d}".format(counter),' '*4,
                          "DLoss: {:.6f}".format(d_loss),' '*4,
                          "GLoss: {:.6f}".format(g_loss),
                           file=log_file)

            epoch_end_time = time.time()
            d_total_loss_list.append(d_loss_list)
            g_total_loss_list.append(g_loss_list)

            d_ave_loss = np.mean(d_loss_list)
            g_ave_loss = np.mean(g_loss_list)
            d_ave_loss_list.append(d_ave_loss)
            g_ave_loss_list.append(g_ave_loss)

            counter  = 0
            SNR_list = []
            print('='*80, file=log_file)
            print(' '*20+'Start Validation for ' + 'Epoch: {0:4d}'.format(epoch), file=log_file)
            for i in idxes_valid:
                counter += 1
                trace_start  = sum(cdpt_max_list[:i])
                trace_end    = trace_start + cdpt_max_list[i]

                inputs_batch = inputs[trace_start:trace_end, self.mute:ns]
                inputs_batch = inputs_batch.reshape(-1, cdpt_max_list[i], ns-self.mute, 1)
                labels_batch = labels[trace_start:trace_end, self.mute:ns]
                labels_batch = labels_batch.reshape(-1, cdpt_max_list[i], ns-self.mute, 1)

                inputs_batch = pre_process(inputs_batch, _prms)
                labels_batch = pre_process(labels_batch, _prms)

                preds_batch   = self.g(inputs_batch, training=False)

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

            self.save_models(epoch, ave_SNR)

            print('='*80, file=log_file)
            print(' '*20+'Epoch: {0:4d}'.format(epoch)+' '*5+'Average Indexes', file=log_file)
            print('Epoch: {:04d}'.format(epoch),' '*7,
                  'Loss: {:.6f}'.format(ave_loss),' '*7,
                  'SNR: {:.6f}'.format(ave_SNR),' '*7,
                  'Time: {:.4f}'.format((epoch_end_time - epoch_start_time) / 3600.0),
                   file=log_file
                 )
            print('='*80, file=log_file)

        img_gan(self.img_dir, d_total_loss_list, g_total_loss_list, d_ave_loss_list, g_ave_loss_list, SNR_ave_list, SNR_total_list, SNR_label_list)
        log_file.close()

        self.test()
        
    def test(self):
        self.restore_models()

        log_path = os.path.join(self.log_dir, 'test_log_'+time.strftime('%Y%m%d_%H%M%S'))
        log_file = open(log_path, 'w')

        inputs, labels, cdpt_max_list, ns, _prms, idxes_train, idxes_valid, idxes = self.dataio.read_data()
        _preds = inputs

        test_start_time = time.time()

        counter  = 0
        SNR_list = []
        SNR_label_list = []
        print('\n'+'='*80, file=log_file)
        print(' '*30+'Start Testing for '+'Epoch: {0:4d}  '.format(self.epoch_start), file=log_file)
        for i in idxes:
            counter += 1

            trace_start  = sum(cdpt_max_list[:i])
            trace_end    = trace_start + cdpt_max_list[i]

            inputs_batch = inputs[trace_start:trace_end, self.mute:ns]
            inputs_batch = inputs_batch.reshape(-1, cdpt_max_list[i], ns-self.mute, 1)
            labels_batch = labels[trace_start:trace_end, self.mute:ns]
            labels_batch = labels_batch.reshape(-1, cdpt_max_list[i], ns-self.mute, 1)

            inputs_batch = pre_process(inputs_batch, _prms)
            labels_batch = pre_process(labels_batch, _prms)

            if self.gan:
                _preds_batch = self.g(inputs_batch, training=False)
            else:
                _preds_batch = self.m(inputs_batch, training=False)

            inputs_batch  = post_process(inputs_batch,  _prms)
            labels_batch  = post_process(labels_batch,  _prms)
            _preds_batch  = post_process(_preds_batch,  _prms)

            _preds[trace_start:trace_end, self.mute:ns] = _preds_batch[0]

            SNR       = self.test_SNR(_preds_batch, labels_batch)
            SNR_label = self.test_SNR(inputs_batch, labels_batch)
            SNR_list.append(SNR)
            SNR_label_list.append(SNR_label)

            if counter % self.test_print_freq == 0:
                print('Counter: {:04d}'.format(counter),' '*20, 
                      'SNR: {:.6f}'.format(SNR),' '*10,
                      'SNR_label: {:.6f}'.format(SNR_label),
                       file=log_file)

        test_end_time = time.time()
        ave_SNR = np.mean(SNR_list)
        ave_SNR_label = np.mean(SNR_label_list)

        self.dataio.write_data(_preds, self.preds_path)

        print('='*80, file=log_file)
        print(' '*32+'Testing Indexes', file=log_file)
        print(' '*10+'Average SNR: {0:.6f},  {1:.6f}'.format(ave_SNR, ave_SNR_label),' '*2,
              'Time: {:.4f} sec'.format((test_end_time - test_start_time)),
               file=log_file)
        print('='*80, file=log_file)

        log_file.close()
