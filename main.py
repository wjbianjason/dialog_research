#-*-coding:utf-8-*-

# 主程序
# 加载各个模块
# （1）预处理对话语料
# （2）加载词向量
# （3）构建tensorflow gragh，单独一个对象
# （4）训练和测试模型

import numpy as np

def tensorflow_main(mode):
	from dialog.dialog_model import DialogModel,ConfigParam
	from dialog.seq2seq_attention_model import ModelParam,Seq2SeqAttentionModel
	from dialog.data_process import Vocab,DataGenerator,Batcher
	#data ready
	_vocab = Vocab(vocab_file="./data/vocab_generate.txt",max_size=10000)
	model_param = ModelParam(mode=mode,batch_size=64,enc_layers=2,num_softmax_samples=0) #most has been set default
	data_generator = DataGenerator(vocab=_vocab,modelParam=model_param,word2vec=None)
	data_generator.loadCorpus("./data/cornell/movie_sample.txt")
	batcher = Batcher(dataGenerator=data_generator, bucketing=True, truncate_input=False)

	config_param = ConfigParam(num_gpus=1) #all has been set default
	dialog = DialogModel(config_param)

	if mode == "train":
		print "begin"
		model = Seq2SeqAttentionModel(model_param,_vocab,num_gpus=config_param.num_gpus)
		dialog.train(model,batcher)
	elif mode == "eval":
		model = Seq2SeqAttentionModel(model_param,_vocab,num_gpus=config_param.num_gpus)
		dialog.eval(model,batcher)
	elif mode == "decode":
		model_param.beam_size = 4
		model_param.batch_size = model_param.beam_size
		config_param.max_decode_steps = 1000000
		config_param.decode_batches_per_ckpt = 8000
		
		decode_param = model_param.copy()
		decode_param.dec_timestemps = 1
		model = Seq2SeqAttentionModel(decode_param,_vocab,num_gpus=config_param.num_gpus)
		dialog.decode(model,batcher)

def keras_main(mode):
	from keras_dialog.sequence_model import SequenceModel
	from keras_dialog.model_factory import ModelFactory
	from dialog.data_process import Vocab
	from keras_dialog.data_process import DataGenerator,ModelParam

	model_param = ModelParam(enc_timesteps=40,dec_timesteps=12,\
		min_input_len=25,min_output_len=5,batch_size=64)
	vocab_in = Vocab("./data/vocab/vocab_input.txt",max_size=36000)
	vocab_out = Vocab("./data/vocab/vocab_output.txt",max_size=18000)
	data_generator = DataGenerator(vocab_in,model_param,vocab_out)
	# data_generator.loadCorpus("./data/train/summary_corpus_train.txt")
	data_generator.loadCorpus("./data/valid/summary_corpus_valid_small.txt")
	valid_data = (np.asarray(data_generator.trainSamples,dtype="float32"),np.asarray(data_generator.trainLabels,dtype="float32"))
	train_samples = 388253
	# count(train) == 388253
	# count(valid) == 6244
 	print "valid samples sum:",data_generator.count

 	# a = data_generator.iterSerializedData("./data/train/summary_corpus.txt")
 	# for i in range(100):
 	# 	print len(a.next()[0])
 	print valid_data[0].shape
	# exit(1)
	if mode == "train":
		model = ModelFactory.get_simple_model(model_param,vocab_in.NumIds(),vocab_out.NumIds(),emb_dims=60)
	# model = ModelFactory.get_attention_model(model_param,vocab_in.NumIds(),vocab_out.NumIds(),emb_dims=20)

		dialog_model = SequenceModel(model = model)
		dialog_model.train_generator(data_generator.iterSerializedData("./data/train/summary_corpus_train.txt"),samples_per_epoch=train_samples,epoch=5,valid_data=valid_data)
		dialog_model.save("keras_attention")
	elif mode == "decode":
		dialog_model = SequenceModel.load("keras_attention")
		test_data = valid_data[0][1]
		print np.stack(test_data)
		result = dialog_model.predict(np.stack(test_data))
		print len(result[0])
		print result[0][1][:30]
		print np.argmax(result[0][1])
		# print vocab_out.IdToWord(6)








if __name__ == "__main__":
	# main("train")
	keras_main("train")

