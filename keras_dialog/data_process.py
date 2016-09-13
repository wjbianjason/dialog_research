#-*-coding:utf-8-*-
# from __future__ import print_function
# import gensim
"""
本文的数据处理主要针对摘要
"""

import nltk
import random
from collections import namedtuple
import numpy as np
from keras.utils import np_utils

ModelParam = namedtuple("ModelParam","enc_timesteps,dec_timesteps,min_input_len,min_output_len,batch_size")


SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'



class DataGenerator(object):
  """Dataset class
  Warning: No vocabulary limit
  """
  def __init__(self,vocab_in,model_param,vocab_out="",word2vec=None):
    """Load all conversations
    """
    self.vocab_in = vocab_in
    if vocab_out == "":
      self.vocab_out = vocab_in
    else:
      self.vocab_out = vocab_out
    self.trainSamples = []  # 2d array containing each question and his answer [[input,target]]
    self.trainLabels = []
    self.word2vec = word2vec
    self._hps = model_param
    self.batch_size = self._hps.batch_size
    self.count = 0

  	
  def iterSerializedData(self,filename):
    while True:
      try:
        Train = []
        Label = []
        count = 0    
        for line in file(filename):
            trainSingleData = line.strip().split("\t\t")
            if len(trainSingleData) != 2:
              continue
            inputData,targetData = self.extractTrainSample(trainSingleData,iter_flag=True)
            if inputData==False and targetData==False:
              continue
            if count < self.batch_size:
              count += 1
              Train.append(inputData)
              Label.append(targetData)
            else:
              yield np.asarray(Train,dtype="float32"),np.asarray(Label,dtype="float32")
              Train = []
              Label = []
              count = 0
      except:
        break 

  def iterData(self):
    while True:
      try:
        Train = []
        Label = []
        count = 0
        for i in range(len(self.trainSamples)):
          inputData = self.trainSamples[i][0]
          targetData = self.trainLabels[i][1]
          if count < self.batch_size:
            count += 1
            Train.append(inputData)
            Label.append(targetData)
          else:
            yield np.asarray(Train),np.asarray(Label)
            Train = []
            Label = []
            count = 0
      except:
        break

  def loadCorpus(self, fileName):
    for line in file(fileName):
        trainSingleData = line.strip().split("\t\t")
        if len(trainSingleData) != 2:
          continue
        self.extractTrainSample(trainSingleData,iter_flag=False)

  def extractTrainSample(self, trainSingleData,truncate_input=False,iter_flag=True):

      # Iterate over all the lines of the conversation
      inputWords  = trainSingleData[0].split()
      targetWords = trainSingleData[1].split()
      if len(inputWords) < self._hps.min_input_len or len(targetWords) < self._hps.min_output_len:
        return False,False
      enc_inputs = []
      dec_inputs = []

      for word in inputWords:
        enc_inputs.append(self.vocab_in.WordToId(word))
      for word in targetWords:
        dec_inputs.append([self.vocab_out.WordToId(word)])
      
      # print len(enc_inputs),len(dec_inputs)

      if not truncate_input:
        if (len(enc_inputs) > self._hps.enc_timesteps or
            len(dec_inputs) > self._hps.dec_timesteps):
          return False,False
      # If we are truncating input, do so if necessary
      else:
        if len(enc_inputs) > self._hps.enc_timesteps:
          enc_inputs = enc_inputs[:self._hps.enc_timesteps]
        if len(dec_inputs) > self._hps.dec_timesteps:
          dec_inputs = dec_inputs[:self._hps.dec_timesteps]

      while len(enc_inputs) < self._hps.enc_timesteps:
        enc_inputs.append(self.vocab_in.WordToId(PAD_TOKEN))
      while len(dec_inputs) < self._hps.dec_timesteps:
        dec_inputs.append([self.vocab_out.WordToId(SENTENCE_END)])

      dec_inputs = np_utils.to_categorical(dec_inputs,self.vocab_out.NumIds())
      # if enc_inputs and dec_inputs:  # Filter wrong samples (if one of the list is empty)
      if iter_flag:
        return enc_inputs,dec_inputs
      else:
        # self.trainSamples.append(enc_inputs)
        # self.trainLabels.append(dec_inputs)
        self.count += 1
        


  def Ids2Words(self,ids_list):
  	"""Get words from ids.
  	Args:
  	ids_list: list of int32
  	vocab: TextVocabulary object
  	Returns:
  	List of words corresponding to ids.
  	"""
  	assert isinstance(ids_list, list), '%s  is not a list' % ids_list
  	return [self.vocab.IdToWord(i) for i in ids_list]

  def shuffle(self):
  	# Shuffle the training samples
  	random.shuffle(self.trainSamples)



if __name__ == "__main__":
  pass

