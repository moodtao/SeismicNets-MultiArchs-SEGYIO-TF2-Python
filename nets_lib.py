import tensorflow as tf
import numpy as np
from typing import List
from functools import partial
from tensorflow import keras
from tensorflow.keras import layers

class ATTN_CAE(keras.Model):
    def __init__(self):
        super(ATTN_CAE, self).__init__()

        self.Conv11 = layers.Conv2D(filters=32, kernel_size=3, padding='same')
        self.Conv12 = layers.Conv2D(filters=32, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))
        self.Conv13 = layers.Conv2D(filters=64, kernel_size=3, padding='same')
        self.Conv14 = layers.Conv2D(filters=64, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))

        self.Conv21 = layers.Conv2D(filters=64, kernel_size=3, padding='same')
        self.Conv22 = layers.Conv2D(filters=64, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))
        self.Conv23 = layers.Conv2D(filters=128, kernel_size=3, padding='same')
        self.Conv24 = layers.Conv2D(filters=128, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))
        
        self.Conv31 = layers.Conv2D(filters=128, kernel_size=3, padding='same')
        self.Conv32 = layers.Conv2D(filters=128, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))
        self.Conv33 = layers.Conv2D(filters=256, kernel_size=3, padding='same')
        self.Conv34 = layers.Conv2D(filters=256, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))
        
        self.Conv41 = layers.Conv2D(filters=256, kernel_size=3, padding='same')
        self.Conv42 = layers.Conv2D(filters=256, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))
        self.Conv43 = layers.Conv2D(filters=128, kernel_size=3, padding='same')
        self.Conv44 = layers.Conv2D(filters=128, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))

        self.Conv51 = layers.Conv2D(filters=128, kernel_size=3, padding='same')
        self.Conv52 = layers.Conv2D(filters=128, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))
        self.Conv53 = layers.Conv2D(filters=64, kernel_size=3, padding='same')
        self.Conv54 = layers.Conv2D(filters=64, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))

        self.Conv61 = layers.Conv2D(filters=64, kernel_size=3, padding='same')
        self.Conv62 = layers.Conv2D(filters=64, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))
        self.Conv63 = layers.Conv2D(filters=32, kernel_size=3, padding='same')
        self.Conv64 = layers.Conv2D(filters=32, kernel_size=1, padding='same', activation=partial(tf.nn.leaky_relu,alpha=0.02))

        self.Conv   = layers.Conv2D(filters=1, kernel_size=3, padding='same')

        self.Atten1 = layers.Conv2D(filters=32, kernel_size=1, strides=1)
        self.Atten2 = layers.Conv2D(filters=32, kernel_size=1, strides=1)
        self.Atten3 = layers.Conv2D(filters=256, kernel_size=1, strides=1)

        self.MaxPool = layers.MaxPooling2D((2, 2), strides=2, padding='same')

        self.gamma = layers.Layer.add_weight(self, name='gamma', shape=[1, ], initializer=tf.initializers.Zeros)

    def call(self, inputs, training=None):
        x = self.Conv11(inputs)
        x = self.Conv12(x)
        x = self.Conv13(x)
        x = self.Conv14(x)
        s01 = np.shape(x)
        x = self.MaxPool(x)

        x = self.Conv21(x)
        x = self.Conv22(x)
        x = self.Conv23(x)
        x = self.Conv24(x)
        s02 = np.shape(x)
        x = self.MaxPool(x)
        
        x = self.Conv31(x)
        x = self.Conv32(x)
        x = self.Conv33(x)
        x = self.Conv34(x)
        s03 = np.shape(x)
        x = self.MaxPool(x)
        s04 = np.shape(x)

        f = self.Atten1(x)
        g = self.Atten2(x)
        h = self.Atten3(x)
        f_flatten = tf.reshape(f, shape=[s04[0], s04[1] * s04[2], 32])
        g_flatten = tf.reshape(g, shape=[s04[0], s04[1] * s04[2], 32])
        h_flatten = tf.reshape(h, shape=[s04[0], s04[1] * s04[2], 256])
        s = tf.matmul(g_flatten, f_flatten, transpose_b=True)
        beta = tf.nn.softmax(s)
        o = tf.matmul(beta, h_flatten) # [bs, N, C]
        gamma = self.gamma
        o = tf.reshape(o, shape=[s04[0], s04[1], s04[2], 256]) # [bs, h, w, C]
        x = gamma * o + x
        
        x = tf.image.resize(x, (s03[1], s03[2]), method='nearest')
        x = self.Conv41(x)
        x = self.Conv42(x)
        x = self.Conv43(x)
        x = self.Conv44(x)
        
        x = tf.image.resize(x, (s02[1], s02[2]), method='nearest')
        x = self.Conv51(x)
        x = self.Conv52(x)
        x = self.Conv53(x)
        x = self.Conv54(x)
        
        x = tf.image.resize(x, (s01[1], s01[2]), method='nearest')
        x = self.Conv61(x)
        x = self.Conv62(x)
        x = self.Conv63(x)
        x = self.Conv64(x)
        
        x = self.Conv(x)

        return x

class ATTN_UNET(keras.Model):
    def __init__(self):
        super(ATTN_UNET, self).__init__()

        self.Conv11 = layers.Conv2D(filters=64, activation='relu', kernel_size=3, padding='same')
        self.Conv12 = layers.Conv2D(filters=64, activation='relu', kernel_size=3, padding='same')

        self.Conv21 = layers.Conv2D(filters=128, activation='relu', kernel_size=3, padding='same')
        self.Conv22 = layers.Conv2D(filters=128, activation='relu', kernel_size=3, padding='same')

        self.Conv31 = layers.Conv2D(filters=256, activation='relu', kernel_size=3, padding='same')
        self.Conv32 = layers.Conv2D(filters=256, activation='relu', kernel_size=3, padding='same')

        self.Conv41 = layers.Conv2D(filters=512, activation='relu', kernel_size=3, padding='same')
        self.Conv42 = layers.Conv2D(filters=512, activation='relu', kernel_size=3, padding='same')

        self.Conv51 = layers.Conv2D(filters=1024, activation='relu', kernel_size=3, padding='same')
        self.Conv52 = layers.Conv2D(filters=1024, activation='relu', kernel_size=3, padding='same')

        self.Trans1 = layers.Conv2DTranspose(512, kernel_size=2, strides=(2, 2), activation='relu')
        self.Conv61 = layers.Conv2D(filters=512, activation='relu', kernel_size=3, padding='same')
        self.Conv62 = layers.Conv2D(filters=512, activation='relu', kernel_size=3, padding='same')

        self.Trans2 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')
        self.Conv71 = layers.Conv2D(filters=256, activation='relu', kernel_size=3, padding='same')
        self.Conv72 = layers.Conv2D(filters=256, activation='relu', kernel_size=3, padding='same')

        self.Trans3 = layers.Conv2DTranspose(128, kernel_size=2, strides=(2, 2), activation='relu')
        self.Conv81 = layers.Conv2D(filters=128, activation='relu', kernel_size=3, padding='same')
        self.Conv82 = layers.Conv2D(filters=128, activation='relu', kernel_size=3, padding='same')

        self.Trans4 = layers.Conv2DTranspose(64, kernel_size=2, strides=(2, 2), activation='relu')
        self.Conv91 = layers.Conv2D(filters=64, activation='relu', kernel_size=3, padding='same')
        self.Conv92 = layers.Conv2D(filters=64, activation='relu', kernel_size=3, padding='same')

        self.Output = layers.Conv2D(1, kernel_size=1)

        self.MaxPool= layers.MaxPooling2D((2, 2), strides=2, padding='valid')

        self.Atten1 = layers.Conv2D(filters=128, kernel_size=1, strides=1)
        self.Atten2 = layers.Conv2D(filters=128, kernel_size=1, strides=1)
        self.Atten3 = layers.Conv2D(filters=1024, kernel_size=1, strides=1)
        
        self.gamma  = layers.Layer.add_weight(self, name='gamma', shape=[1, ], initializer=tf.initializers.Zeros)

    def call(self, inputs, training=None):

        x   = self.Conv11(inputs)
        x04 = self.Conv12(x)
        s04 = np.shape(x04)
        x   = self.MaxPool(x04)
        x   = self.Conv21(x)
        x03 = self.Conv22(x)
        s03 = np.shape(x03)
        x   = self.MaxPool(x03)
        x   = self.Conv31(x)
        x02 = self.Conv32(x)
        s02 = np.shape(x02)
        x   = self.MaxPool(x02)
        x   = self.Conv41(x)
        x01 = self.Conv42(x)
        s01 = np.shape(x01)
        x   = self.MaxPool(x01)
        x   = self.Conv51(x)
        x   = self.Conv52(x)
        s00 = np.shape(x)

        f = self.Atten1(x)
        g = self.Atten2(x)
        h = self.Atten3(x)
        f_flatten = tf.reshape(f, shape=[s00[0], s00[1] * s00[2], 128])
        g_flatten = tf.reshape(g, shape=[s00[0], s00[1] * s00[2], 128])
        h_flatten = tf.reshape(h, shape=[s00[0], s00[1] * s00[2], 1024])
        s = tf.matmul(g_flatten, f_flatten, transpose_b=True)
        beta = tf.nn.softmax(s)
        o = tf.matmul(beta, h_flatten) # [bs, N, C]
        gamma = self.gamma
        o = tf.reshape(o, shape=[s00[0], s00[1], s00[2], 1024]) # [bs, h, w, C]
        x = gamma * o + x

        x11 = self.Trans1(x)
        x11 = tf.pad(x11, [[0,0],[s01[1]-s00[1]*2,0],[s01[2]-s00[2]*2,0],[0,0]], 'CONSTANT')

        x   = layers.concatenate([x11, x01], axis=-1)
        
        x   = self.Conv61(x)
        x   = self.Conv62(x)
        s00 = np.shape(x)

        x12 = self.Trans2(x)
        x12 = tf.pad(x12, [[0,0],[s02[1]-s00[1]*2,0],[s02[2]-s00[2]*2,0],[0,0]], 'CONSTANT')
        x   = layers.concatenate([x12, x02], axis=-1)
        
        x   = self.Conv71(x)
        x   = self.Conv72(x)
        s00 = np.shape(x)

        x13 = self.Trans3(x)
        x13 = tf.pad(x13, [[0,0],[s03[1]-s00[1]*2,0],[s03[2]-s00[2]*2,0],[0,0]], 'CONSTANT')
        x   = layers.concatenate([x13, x03], axis=-1)
        
        x   = self.Conv81(x)
        x   = self.Conv82(x)
        s00 = np.shape(x)

        x14 = self.Trans4(x)
        x14 = tf.pad(x14, [[0,0],[s04[1]-s00[1]*2,0],[s04[2]-s00[2]*2,0],[0,0]], 'CONSTANT')
        x   = layers.concatenate([x14, x04], axis=-1)
        
        x   = self.Conv91(x)
        x   = self.Conv92(x)

        x   = self.Output(x)

        return x
