# -*- coding: utf-8 -*-
# seq2seq_attention: 1.word embedding 2.encoder 3.decoder(optional with attention). for more detail, please check:Neural Machine Translation By Jointly Learning to Align And Translate
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
import random
import copy
from smiles2vec_v3_young import smiles2vec

def index2onehot(infile, vocab_size):
    d1,d2=np.shape(infile) # 13642,100
    all_data=[]
    for i in range(d1):
        one_smiles=[]
        for j in range(d2):
            onehot=np.zeros([1,1,vocab_size])
            onehot[0,0,infile[i,j]]=1.0
            one_smiles.append(onehot)
        one_feature=np.concatenate(one_smiles,axis=1)
        all_data.append(one_feature)
    all_feature=np.array(np.concatenate(all_data,axis=0))
    return all_feature
        



#  started: learn to output reverse sequence of itself.
def main():
    #1. Load Data: Soo Rewrite
    trainX, testX, trainY_raw ,testY_raw  = smiles2vec("./CIF_to_labels.txt")
    vocab_size=25
    #((13642, 100), (1516, 100), (13642, 1), (1516, 1), 24)
    #2. convert index to one 1 embedding
    trainX=index2onehot(trainX,vocab_size) 
    testX=index2onehot(testX,vocab_size)   
    
    #Placeholder
    # Specify feature
    feature_columns = [tf.feature_column.numeric_column("x", shape=[100, 25])]

    batch_size = 500
    # Build 4 layer DNN classifier
    estimator = tf.estimator.DNNRegressor(
        feature_columns=feature_columns,
        dropout=0.5,
        hidden_units=[1024, 512, 256],
        optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": trainX},
        y=np.array(trainY_raw),
        num_epochs=None,
        batch_size=500,
        shuffle=True
    )

    
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": testX},
        y=np.array(testY_raw),
        num_epochs=1,
        shuffle=False
    )

    #Evaluaate accuracy
    for i in xrange(1,1000000):
        num=int(1+i*100)
        estimator.train(input_fn=train_input_fn,steps=num)
        accuracy_score = estimator.evaluate(input_fn=test_input_fn)
        print("step "+str(num)+" Loss: {0:f}\n".format(accuracy_score["loss"]))

        if 1==0: #i>0 and i%100==0: # let's not record log
            predictions = estimator.predict(input_fn=test_input_fn)
            for i , j in zip(predictions,range(len(testY_raw))):
                print(i,"\n\n\n\n",testY_raw[j][0])
main()
