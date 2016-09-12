#-*-coding:utf-8-*-
# from __future__ import print_function
import sys
# import gensim
import numpy as np
import os
import tensorflow as tf
import nltk
import random
import Queue
from threading import Thread
import time
from collections import namedtuple

# from dialog.uitl import Util
from dialog.seq2seq_attention_model import ModelParam

# Special tokens
# PARAGRAPH_START = '<p>'
# PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
BUCKET_CACHE_BATCH = 100
QUEUE_NUM_BATCH = 100
# DOCUMENT_START = '<d>'
# DOCUMENT_END = '</d>'

ModelInput = namedtuple('ModelInput',
                        'enc_input dec_input target enc_len dec_len '
                        'origin_article origin_abstract')


class Vocab(object):
  """Vocabulary class for mapping words and ids."""

  def __init__(self, vocab_file, max_size,min_fre=5):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0
    for word in [SENTENCE_START,SENTENCE_END,UNKNOWN_TOKEN,PAD_TOKEN]:
      self.CreateWord(word)
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          sys.stderr.write('Bad line: %s\n' % line)
          continue
        if pieces[0] in self._word_to_id:
          raise ValueError('Duplicated word: %s.' % pieces[0])
        self._word_to_id[pieces[0]] = self._count
        self._id_to_word[self._count] = pieces[0]
        self._count += 1
        if self._count > max_size-1:
          sys.stderr.write('Too many words: >%d.' % max_size)
          break


  def WordToId(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def IdToWord(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('id not found in vocab: %d.' % word_id)
    return self._id_to_word[word_id]

  def NumIds(self):
    return self._count

  def CreateWord(self,word):
  	if word not in self._word_to_id:
		self._word_to_id[word] = self._count
		self._id_to_word[self._count] = word
		self._count += 1

class Word2Vec(object):
    BlankVector = np.zeros((1, 300))
    def __init__(self, path='../data/GoogleNews-vectors-negative300.bin'):
        import gensim
        if not os.path.isfile(path):
            raise Exception('Word2Vec file ' + path + ' not found')
        print('Initializing Word2Vec from file ', path)
        self.model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
        print('Initialization complete')

    def get_vector(self, word):
        """
        Gets vector for particular word. If not found returns the UnknownVector constant

        :type word: str
        :param word: the word for which vector is requested
        :rtype: np.array
        """
        if word in self.model:
            return self.model[word]
        else:
            return Word2Vec.BlankVector

    def get_words(self, vector, n=10):
        return self.model.similar_by_vector(vector, topn=n)[0][0]

    def get_top_word(self, vector):
        """
        if vector is all 0 returns blank  string, if vector is all 1 returns Unknown word
        :param vector: np.array of dimension Word2VecDimension
        :return: word
        """
        if np.all([(i == 0) for i in vector]):
            return '<Blank>'
        return self.get_words(vector, 10)

class DataGenerator(object):
  """Dataset class
  Warning: No vocabulary limit
  """
  def __init__(self, vocab, modelParam,word2vec=None):
  	"""Load all conversations
  	"""
  	self.vocab = vocab
  	self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]
  	self.originData = []
  	self._hps = modelParam
  	self.word2vec = word2vec
	self.SENTENCE_START = vocab.WordToId(SENTENCE_START)
    	self.SENTENCE_END = vocab.WordToId(SENTENCE_END)
    	self.PAD_TOKEN = vocab.WordToId(PAD_TOKEN)
    	self.UNKNOWN_TOKEN = vocab.WordToId(UNKNOWN_TOKEN)
  	# self.inputMaxLen = modelParam.enc_timesteps
  	# self.targetMaxlen = modelParam.dec_timesteps
  	
  def iterSerializedData(self,filename):
    for line in file(filename):
        trainSingleData = line.strip().split("\t")
        if len(trainSingleData) != 2:
          continue
        inputWords  = self.extractText(trainSingleData[0])
        targetWords = self.extractText(trainSingleData[1], True)
        if inputWords and targetWords:
          yield (inputWords,targetWords,trainSingleData[0],trainSingleData[1])


  def iterData(self):
    while True:
      try:
      	for i in range(len(self.trainingSamples)):
      		inputData = self.trainingSamples[i][0]
      		targetData = self.trainingSamples[i][1]
      		originInput = self.originData[i][0]
      		originTarget = self.originData[i][1]
      		yield (inputData,targetData,originInput,originTarget)
      except:
        break
      

  def getWordId(self,word):
  	# word = word.lower() # or not lower
  	return self.vocab.WordToId(word)

  def loadCorpus(self, fileName):
    # cornellData = CornellData(fileName)
    # conversations = cornellData.getConversations()
    # trainRawData = cornellData.getTrainRawData()
    # for trainSingleData in trainRawData:
    for line in file(fileName):
        trainSingleData = line.strip().split("\t")
        if len(trainSingleData) != 2:
          continue
        self.extractTrainSample(trainSingleData)
    print('Loaded: {} words, {} QA'.format(self.vocab.NumIds(), len(self.trainingSamples)))

  def extractTrainSample(self, trainSingleData):
      """Extract the sample lines from the conversations
      Args:
          conversation (Obj): a convesation object containing the lines to extract
      """
      # Iterate over all the lines of the conversation
      inputWords  = self.extractText(trainSingleData[0])
      targetWords = self.extractText(trainSingleData[1], True)
      if inputWords and targetWords:  # Filter wrong samples (if one of the list is empty)
          self.trainingSamples.append([inputWords, targetWords])
          self.originData.append([trainSingleData[0],trainSingleData[1]])


  def extractText(self, line, isTarget=False):
      """Extract the words from a sample lines
      Args:
          line (str): a line containing the text to extract
          isTarget (bool): Define the question on the answer
      Return:
          list<int>: the list of the word ids of the sentence
      """
      #对于对话，进行了只取一句的处理

      words = []
      # Extract sentences
      try:
        sentencesToken = nltk.sent_tokenize(line)
      except:
        return False
      # We add sentence by sentence until we reach the maximum length
      for i in range(len(sentencesToken)):
          # If question: we only keep the last sentences
          # If answer: we only keep the first sentences
          maxLength = self._hps.dec_timesteps -1 # descrease 1 because add the special target
          if not isTarget:
              i = len(sentencesToken)-1 - i
              maxLength = self._hps.enc_timesteps
          tokens = nltk.word_tokenize(sentencesToken[i])
          # If the total length is not too big, we still can add one more sentence
          if len(words) + len(tokens) <= maxLength:
              tempWords = []
              for token in tokens:
                  tempWords.append(self.getWordId(token))  # Create the vocabulary and the training sentences
              if isTarget:
                  words = words + tempWords
              else:
                  words = tempWords + words
          else:
              break  # We reach the max length already
      return words

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
  	random.shuffle(self.trainingSamples)

class Batcher(object):
  """Batch reader with shuffling and bucketing support."""

  def __init__(self, dataGenerator, bucketing=True, truncate_input=False):
    """Batcher constructor.

    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary.
      hps: Seq2SeqAttention model hyperparameters.
      article_key: article feature key in tf.Example.
      abstract_key: abstract feature key in tf.Example.
      max_article_sentences: Max number of sentences used from article.
      max_abstract_sentences: Max number of sentences used from abstract.
      bucketing: Whether bucket articles of similar length into the same batch.
      truncate_input: Whether to truncate input that is too long. Alternative is
        to discard such examples.
    """
    self._data_generator = dataGenerator
    self._vocab = dataGenerator.vocab
    self._hps =  dataGenerator._hps 

    # self._max_article_sentences = self.
    # self._max_abstract_sentences = max_abstract_sentences
    self._bucketing = bucketing
    self._truncate_input = truncate_input
    self._input_queue = Queue.Queue(QUEUE_NUM_BATCH * self._hps.batch_size)
    self._bucket_input_queue = Queue.Queue(QUEUE_NUM_BATCH)
    self._input_threads = []
    for _ in xrange(8):
      self._input_threads.append(Thread(target=self._FillInputQueue))
      self._input_threads[-1].daemon = True
      self._input_threads[-1].start()
    self._bucketing_threads = []
    for _ in xrange(2):
      self._bucketing_threads.append(Thread(target=self._FillBucketInputQueue))
      self._bucketing_threads[-1].daemon = True
      self._bucketing_threads[-1].start()

    # self._watch_thread = Thread(target=self._WatchThreads)
    # self._watch_thread.daemon = True
    # self._watch_thread.start()

  def NextBatch(self):
    """Returns a batch of inputs for seq2seq attention model.

    Returns:
      enc_batch: A batch of encoder inputs [batch_size, hps.enc_timestamps].
      dec_batch: A batch of decoder inputs [batch_size, hps.dec_timestamps].
      target_batch: A batch of targets [batch_size, hps.dec_timestamps].
      enc_input_len: encoder input lengths of the batch.
      dec_input_len: decoder input lengths of the batch.
      loss_weights: weights for loss function, 1 if not padded, 0 if padded.
      origin_articles: original article words.
      origin_abstracts: original abstract words.
    """
    enc_batch = np.zeros(
        (self._hps.batch_size, self._hps.enc_timesteps), dtype=np.int32)
    enc_input_lens = np.zeros(
        (self._hps.batch_size), dtype=np.int32)
    dec_batch = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
    dec_output_lens = np.zeros(
        (self._hps.batch_size), dtype=np.int32)
    # label
    target_batch = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
    loss_weights = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.float32)
    origin_articles = ['None'] * self._hps.batch_size
    origin_abstracts = ['None'] * self._hps.batch_size

    buckets = self._bucket_input_queue.get()
    for i in xrange(self._hps.batch_size):
      (enc_inputs, dec_inputs, targets, enc_input_len, dec_output_len,
       article, abstract) = buckets[i]

      origin_articles[i] = article
      origin_abstracts[i] = abstract
      enc_input_lens[i] = enc_input_len
      dec_output_lens[i] = dec_output_len
      enc_batch[i, :] = enc_inputs[:]
      dec_batch[i, :] = dec_inputs[:]
      target_batch[i, :] = targets[:]
      for j in xrange(dec_output_len):
        loss_weights[i][j] = 1
    return (enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens,
            loss_weights, origin_articles, origin_abstracts)

  def _FillInputQueue(self):
    """Fill input queue with ModelInput."""
    start_id = self._vocab.WordToId(SENTENCE_START)
    end_id = self._vocab.WordToId(SENTENCE_END)
    pad_id = self._vocab.WordToId(PAD_TOKEN)
    input_gen = self._data_generator.iterData()
    while True:
      (inputData, targetData,originInput,originTarget) = input_gen.next()
      enc_inputs = []
      # Use the <s> as the <GO> symbol for decoder inputs.
      dec_inputs = [start_id]
      enc_inputs += inputData
      dec_inputs += targetData

      # Filter out too-short input
      if (len(enc_inputs) < self._hps.min_input_len or
          len(dec_inputs) < self._hps.min_input_len):
        # tf.logging.warning('Drop an example - too short.\nenc:%d\ndec:%d',
                           # len(enc_inputs), len(dec_inputs))
        continue
      # If we're not truncating input, throw out too-long input
      if not self._truncate_input:
        if (len(enc_inputs) > self._hps.enc_timesteps or
            len(dec_inputs) > self._hps.dec_timesteps):
          tf.logging.warning('Drop an example - too long.\nenc:%d\ndec:%d',
                             len(enc_inputs), len(dec_inputs))
          continue
      # If we are truncating input, do so if necessary
      else:
        if len(enc_inputs) > self._hps.enc_timesteps:
          enc_inputs = enc_inputs[:self._hps.enc_timesteps]
        if len(dec_inputs) > self._hps.dec_timesteps:
          dec_inputs = dec_inputs[:self._hps.dec_timesteps]

      # targets is dec_inputs without <s> at beginning, plus </s> at end
      targets = dec_inputs[1:]
      targets.append(end_id)

      # Now len(enc_inputs) should be <= enc_timesteps, and
      # len(targets) = len(dec_inputs) should be <= dec_timesteps

      enc_input_len = len(enc_inputs)
      dec_output_len = len(targets)

      # Pad if necessary
      while len(enc_inputs) < self._hps.enc_timesteps:
        enc_inputs.append(pad_id)
      while len(dec_inputs) < self._hps.dec_timesteps:
        dec_inputs.append(end_id)
      while len(targets) < self._hps.dec_timesteps:
        targets.append(end_id)

      element = ModelInput(enc_inputs, dec_inputs, targets, enc_input_len,
                           dec_output_len, originInput,
                           originTarget)
      self._input_queue.put(element)

  def _FillBucketInputQueue(self):
    """Fill bucketed batches into the bucket_input_queue."""
    while True:
      inputs = []
      for _ in xrange(self._hps.batch_size * BUCKET_CACHE_BATCH):
        inputs.append(self._input_queue.get())
      if self._bucketing:
        inputs = sorted(inputs, key=lambda inp: inp.enc_len)

      batches = []
      for i in xrange(0, len(inputs), self._hps.batch_size):
        batches.append(inputs[i:i+self._hps.batch_size])
      random.shuffle(batches)
      for b in batches:
        self._bucket_input_queue.put(b)

  def _WatchThreads(self):
    """Watch the daemon input threads and restart if dead."""
    while True:
      time.sleep(60)
      input_threads = []
      for t in self._input_threads:
        if t.is_alive():
          input_threads.append(t)
        else:
          tf.logging.error('Found input thread dead.')
          new_t = Thread(target=self._FillInputQueue)
          input_threads.append(new_t)
          input_threads[-1].daemon = True
          input_threads[-1].start()
      self._input_threads = input_threads

      bucketing_threads = []
      for t in self._bucketing_threads:
        if t.is_alive():
          bucketing_threads.append(t)
        else:
          tf.logging.error('Found bucketing thread dead.')
          new_t = Thread(target=self._FillBucketInputQueue)
          bucketing_threads.append(new_t)
          bucketing_threads[-1].daemon = True
          bucketing_threads[-1].start()
      self._bucketing_threads = bucketing_threads


if __name__ == "__main__":
  # Util.generate_vocab("../data/cornell/movie_conv.txt",min_fre=5)
  _vocab = Vocab(vocab_file="../data/vocab_generate.txt",max_size=10000)
  print _vocab.NumIds()
  model_param = ModelParam(mode="train",batch_size=64) #most has been set default
  print model_param.__dict__
  data_generator = DataGenerator(vocab=_vocab,modelParam=model_param)
  print model_param.dec_timesteps
  data_generator.loadCorpus("../data/cornell/movie_sample.txt")
  batcher = Batcher(dataGenerator=data_generator,bucketing=True,truncate_input=False)
  print batcher.NextBatch()

  # a = data_generator.iterData()
  # while True:
  #   data = a.next()
  #   print data


