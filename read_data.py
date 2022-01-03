import os
import re
import sys
import math
import segyio
import numpy as np
import tensorflow as tf
from shutil import copyfile


class DATAIO():
    def __init__(self, inputs_path, labels_path, preds_file, batch_size, valid_ratio):
        self.inputs_path = inputs_path
        self.labels_path = labels_path
        self.preds_file  = preds_file
        self.batch_size  = batch_size
        self.valid_ratio = valid_ratio
        self.trace_count = 0
        self.ns          = 0

    def read_data(self):

        inputs_file = segyio.open(self.inputs_path, ignore_geometry=True)
        labels_file = segyio.open(self.labels_path, ignore_geometry=True)

        inputs = inputs_file.trace.raw[:]
        labels = labels_file.trace.raw[:]

        if np.shape(inputs)[0] != np.shape(labels)[0]:
            print('Mismatched Train Datasets!!')
            sys.exit()

        #trace count
        self.trace_count = np.shape(inputs)[0]
        #sampling points
        self.ns = np.shape(inputs)[1]

        #cdpt max
        cdpt_list = []
        cdpt_key  = re.compile(r'(?<=CDP_TRACE: )\d+')
        for trace in np.arange(inputs_file.tracecount):
            cdpt_list.append(int(cdpt_key.findall(str(inputs_file.header[trace].values()))[0]))
        cdpt_list.append(-1e10)

        cdpt_max_list = []
        for i in np.arange(len(cdpt_list)-1):
            if cdpt_list[i] > cdpt_list[i+1]:
                cdpt_max_list.append(cdpt_list[i])
            else:
                pass

        total_batch = len(cdpt_max_list) // self.batch_size
        print('Trace Count: {0:6d}, Ns: {1:6d}, Total Batch: {2:6d}'.format(self.trace_count, self.ns, total_batch))

        idxes = np.arange(int(total_batch))

        inputs = inputs.reshape((self.trace_count, self.ns, 1))
        labels = labels.reshape((self.trace_count, self.ns, 1))

        _prms = [];
        _prms.append(np.mean(inputs))
        _prms.append(max(np.std(inputs), 1e-4))

        _batch = int(int(total_batch) * self.valid_ratio)
        idxes_valid = np.ceil(np.linspace(0, total_batch - 1, _batch)).astype(int)
        idxes_train = np.array(list(set(idxes).difference(set(idxes_valid))))
        np.random.shuffle(idxes_train)

        inputs_file.close()
        labels_file.close()

        return inputs, labels, cdpt_max_list, self.ns, _prms, idxes_train, idxes_valid, idxes

    def write_data(self, preds, preds_path):
        copyfile(self.inputs_path, preds_path)

        preds_file = segyio.open(preds_path, 'r+', ignore_geometry=True)
        preds = preds.reshape((self.trace_count, self.ns))
        preds_file.trace.raw[:] = preds

        preds_file.close()

        return print('Predictions Saved!')
