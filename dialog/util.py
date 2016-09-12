#-*-coding:utf-8-*-
import tensorflow as tf
import nltk
import struct
from tensorflow.core.example import example_pb2

class Util(object):
  """some util function"""
  @staticmethod
  def generate_vocab(filename,min_fre=5,prefix=""):
    vf = open("../data/"+prefix+"vocab_generate.txt",'w')
    word = {}
    for line in file(filename):
      line = line.strip()
      try:
        sentencesToken = nltk.sent_tokenize(line)
      except:
        continue
      for i in range(len(sentencesToken)):
          tokens = nltk.word_tokenize(sentencesToken[i])
          for token in tokens:
              word.setdefault(token,0)
              word[token] += 1
    for char,num in sorted(word.items(),key=lambda x:x[1],reverse=True):
      if num < min_fre:
        break
      vf.write(char+" "+str(num)+"\n")

  @staticmethod
  def generate_normal_vocab(filename,min_fre=10,output="vocab_generate.txt"):
  	vf = open("../data/"+output,'w')
  	wordDic = {}
  	for line in file(filename):
  		line = line.strip().split("\t\t")[0]
  		words = line.strip().split()
  		for word in words:
  			wordDic.setdefault(word,0)
  			wordDic[word] += 1
	for char,num in sorted(wordDic.items(),key=lambda x:x[1],reverse=True):
		if num < min_fre:
			break
		vf.write(char+" "+str(num)+"\n")



  @staticmethod
  def generate_new_copurs(fileTitle,fileArticle):
  	vocab = []
  	for line in file("../data/vocab_output.txt"):
  		units = line.strip().split()
  		vocab.append(units[0])
  	fArt = open(fileArticle,'r')
  	fCorpus = open("../data/summary_corpus.txt",'w')
  	for line in file(fileTitle):
		lineArticle = fArt.readline()
		newLine = line.strip()
		titleWords = newLine.split()
		flag = True
		for word in titleWords:
			if word not in vocab:
				flag = False
				# print word
		if flag:
			fCorpus.write(lineArticle.strip()+"\t\t"+line)

  	


  def recordWriter(self,inputData,label,filename):
  	writer = tf.python_io.TFRecordWriter(filename)
  	for i in range(len(inputData)):
  		x = inputData[i]
  		y = label[i]
  		example = tf.train.Example(
  	  		features = tf.train.Features(
  	   		 feature = {
  	      		'inputData': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x])),
  	      		'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y]))
  	    }
  	  )
  	)
  		serialized = example.SerializeToString()
  		writer.write(serialized)
  	writer.close()

	def TextGenerator(self, filename):
		example_gen = self._ExampleGen(filename)
		while True:
			e = example_gen.next()
			try:
				inputData = self._GetExFeatureText(e, "inputData")
				label = self._GetExFeatureText(e, "label")
			except ValueError:
				tf.logging.error('Failed to get data from file')
		    	continue
		  	yield (inputData, label)

	def _ExampleGen(self,filename):
		reader = open(filename, 'rb')
		while True:
			len_bytes = reader.read(8)
			if not len_bytes: break
			str_len = struct.unpack('q', len_bytes)[0]
			example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
			yield example_pb2.Example.FromString(example_str)

	def _GetExFeatureText(self, ex, key):
		return ex.features.feature[key].bytes_list.value[0]


if __name__ == "__main__":
	# Util.generate_normal_vocab("/home/bianweijie/dataset/keras.title.txt",prefix='keras')
	Util.generate_normal_vocab("../data/summary_corpus.txt",min_fre=5,output="vocab_input.txt")
	# Util.generate_new_copurs("/home/bianweijie/dataset/keras.title.txt","/home/bianweijie/dataset/keras.article.txt")

