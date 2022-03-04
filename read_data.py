import os
import re
import sys
import math
import segyio
import numpy as np
import tensorflow as tf
from shutil import copyfile
from collections import Counter


class DATAIO():
    def __init__(self, inputs_path, labels_path, tests_path, preds_path, valid_ratio):
        self.inputs_path   = inputs_path
        self.labels_path   = labels_path
        self.tests_path    = tests_path
        self.preds_path    = preds_path
        self.valid_ratio   = valid_ratio
        self.trace_count   = 0
        self.ns            = 0
        self.cdpt_max_list = []
        self.inputs_file   = 0
        self.spec          = 0
        self.labels_file   = 0

    def read_data(self):
        self.inputs_file   = segyio.open(self.inputs_path, ignore_geometry=True)
        self.labels_file   = segyio.open(self.labels_path, ignore_geometry=True)
        if self.inputs_file.tracecount != self.labels_file.tracecount:
            print('Mismatched Train Datasets!!')
            sys.exit()

        #trace count
        self.trace_count = self.inputs_file.tracecount
        #sampling points
        self.ns = self.inputs_file.bin[segyio.BinField.Samples]

        #cdpt max
        CDP                = self.inputs_file.attributes(segyio.TraceField.CDP)[:]
        CDP_counter        = dict( Counter(CDP) )
        self.cdpt_max_list = list( CDP_counter.values() )
        total_batch        = len(self.cdpt_max_list)
        print('Trace Count: {0:6d}, Ns: {1:6d}, Total Batch: {2:6d}'.format(self.trace_count, self.ns, total_batch))

        idxes = np.arange(int(total_batch))
        _batch = int(int(total_batch) * self.valid_ratio)
        idxes_valid = np.ceil(np.linspace(0, total_batch - 1, _batch)).astype(int)
        idxes_train = np.array(list(set(idxes).difference(set(idxes_valid))))
        np.random.shuffle(idxes_train)

        _prms = [];
        _prms.append(np.mean(self.inputs_file.trace.raw[:int(self.trace_count*0.001)]))
        _prms.append(max(np.std(self.inputs_file.trace.raw[:int(self.trace_count*0.001)]), 1e-4))

        return _prms, idxes_train, idxes_valid

    def read_inputs(self):
        self.inputs_file   = 0
        self.cdpt_max_list = []

        self.inputs_file   = segyio.open(self.tests_path, ignore_geometry=True)
        self.spec          = segyio.tools.metadata(self.inputs_file)
        #trace count
        self.trace_count = self.inputs_file.tracecount
        #sampling points
        self.ns = self.inputs_file.bin[segyio.BinField.Samples]

        #cdpt max
        CDP                = self.inputs_file.attributes(segyio.TraceField.CDP)[:]
        CDP_counter        = dict( Counter(CDP) )
        self.cdpt_max_list = list( CDP_counter.values() )
        total_batch        = len(self.cdpt_max_list)
        print('Trace Count: {0:6d}, Ns: {1:6d}, Test Total Batch: {2:6d}'.format(self.trace_count, self.ns, total_batch))

        idxes = np.arange(int(total_batch))

        _prms = [];
        _prms.append(np.mean(self.inputs_file.trace.raw[:int(self.trace_count*0.001)]))
        _prms.append(max(np.std(self.inputs_file.trace.raw[:int(self.trace_count*0.001)]), 1e-4))

        return _prms, idxes

    def copy_(self):
        return self.inputs_file, self.spec

    def loc_(self, i):
        trace_start  = sum(self.cdpt_max_list[:i])
        trace_end    = trace_start + self.cdpt_max_list[i]
        return trace_start, trace_end        

    def ret_train_gather(self, i, mute):
        trace_start, trace_end = self.loc_(i)
        inputs = self.inputs_file.trace.raw[trace_start:trace_end][:,mute:self.ns]
        inputs = inputs.reshape(1, self.cdpt_max_list[i], self.ns-mute, 1)
        labels = self.labels_file.trace.raw[trace_start:trace_end][:,mute:self.ns]
        labels = labels.reshape(1, self.cdpt_max_list[i], self.ns-mute, 1)
        return inputs, labels

    def ret_test_gather(self, i, mute):
        trace_start, trace_end = self.loc_(i)
        inputs = self.inputs_file.trace.raw[trace_start:trace_end][:,mute:self.ns]
        inputs = inputs.reshape(1, self.cdpt_max_list[i], self.ns-mute, 1)
        return inputs

    def write_gather(self, i, _preds, _file, mute):
        trace_start, trace_end = self.loc_(i)
        gather       = np.zeros(np.shape(_preds))
        gather[0]    = _preds[0]
        _gather      = self.inputs_file.trace.raw[trace_start:trace_end]
        _gather[:][:,mute:self.ns] = gather.reshape(self.cdpt_max_list[i], self.ns-mute)
        _file.trace.raw[trace_start:trace_end] = _gather

    def close_files(self):
        self.inputs_file.close()
        self.labels_file.close()
