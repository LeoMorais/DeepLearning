# -*- Mode: Python; coding: utf-8 -*-

"""
Created on Sat Apr 29 15:54:45 2017

@author: www.deeplearningbrasil.com.br
"""

# Rede Neural convolucional simples para o problema de reconhecimento de d�gitos (MNIST)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import random
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import time
start_time = time.time()

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Verifique o site https://www.tensorflow.org/get_started/mnist/beginners para
# mais informa��es sobre o conjunto de dados

# Par�metros de aprendizagem
taxa_aprendizado = 0.001
quantidade_maxima_epocas = 50
batch_size = 100
Y_conv = 2

# entrada dos place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # imagem 28x28x1 (preto e branca)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 512], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

#L2_flat = tf.reshape(L2, [-1, 7 * 7 * 2048])
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''
# L3 ImgIn shape=(?, 14, 14, 32)
W3 = tf.Variable(tf.random_normal([3, 3, 512, 1024], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

W4 = tf.Variable(tf.random_normal([3, 3, 1024, 2048], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')


L4_flat = tf.reshape(L4, [-1, 2 * 2 * 2048])

# Classificador - Camada Fully Connected entrada 7x7x64 -> 10 sa�das
W5 = tf.get_variable("W5", shape=[2 * 2 * 2048, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4_flat, W5) + b

# Define a fun��o de custo e o m�todo de otimiza��o 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#optimizer = tf.train.AdamOptimizer(taxa_aprendizado).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(taxa_aprendizado).minimize(cost)

# inicializa
config = tf.ConfigProto(
    log_device_placement=True, allow_soft_placement=True
)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# treina a rede
xplot = []
yplot = []
print('Rede inicialiada. Treinamento inicializado. Tome um cafe...')
for epoca in range(quantidade_maxima_epocas):
    custo_medio = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        custo_medio += c / total_batch     
    print('Epoca:', '%04d' % (epoca + 1), 'perda =', '{:.9f}'.format(custo_medio))
    xplot.append(epoca)
    yplot.append('{:.4f}'.format(custo_medio))
    plt.plot(xplot, yplot)
    
print('Treinamento finalizado!')

#Teste o modelo e verifica a taxa de acerto
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
ac = []
for i in range(total_batch):
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    ac.append(sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys}))
soma = 0
for j in range(len(ac)):
    soma = soma + ac[j]
acuracia = soma/len(ac)

#Obt�m uma nova imagem e testa o modelo
r = random.randint(0, mnist.test.num_examples - 1)
classe = sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1))
predicao = sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]})


#Saída da rede
#d = tf.cast(tf.argmax(logits, 1),tf.float32)
#Respostas Corretas
#y = tf.cast(tf.argmax(Y,1),tf.float32)
 
#Calcula a área da curva ROC (AUC)
#auc, update_auc = tf.contrib.metrics.streaming_auc(d,y)



#Output resultados
print('#####################################################')
print('Taxa de acerto:', acuracia)
print("Classe real: ", classe)
print("Predicao: ", predicao)
print("--- %s seconds ---" % (time.time() - start_time))
#print('Area da curva ROC:', sess.run(update_auc, feed_dict={
#      X: mnist.test.images, Y: mnist.test.labels}))

fim = start_time - time.time()
print(fim)
plt.show()
print(xplot,yplot)
