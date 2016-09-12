#-*-coding:utf-8-*-

# 模型参数类
# 加载模型
# 加载数据
# 开启训练或测试

# from __future__ import print_function
import tensorflow as tf
import sys
import time
from dialog.seq2seq_attention_decode import BSDecoder

class ConfigParam(object):
	def __init__(self,num_gpus=0,train_dir="./data",log_root="./data",checkpoint_secs=60,\
		max_run_steps=10000000,eval_dir="./data",eval_interval_secs=60,decode_dir="./data",max_decode_steps=1000000,decode_batches_per_ckpt=8000):
		self.num_gpus = num_gpus
		self.train_dir = train_dir
		self.log_root = log_root
		self.checkpoint_secs = checkpoint_secs
		self.max_run_steps = max_run_steps
		self.eval_dir = eval_dir
		self.eval_interval_secs = eval_interval_secs
		self.decode_dir = decode_dir
		self.max_decode_steps = max_decode_steps
		self.decode_batches_per_ckpt = decode_batches_per_ckpt



class DialogModel:
	def __init__(self,configParam):
		self._config_params = configParam

	def train(self,model,data_batcher):
	 	"""Runs model training."""
		with tf.device('/cpu:0'):
			model.build_graph()
			saver = tf.train.Saver()
			# Train dir is different from log_root to avoid summary directory
			# conflict with Supervisor.
			summary_writer = tf.train.SummaryWriter(self._config_params.train_dir)
			sv = tf.train.Supervisor(logdir=self._config_params.log_root,
			                         is_chief=True,
			                         saver=saver,
			                         summary_op=None,
			                         save_summaries_secs=60,
			                         save_model_secs=self._config_params.checkpoint_secs,
			                         global_step=model.global_step)
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
		    	sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
			running_avg_loss = 0
			step = 0
			while not sv.should_stop() and step < self._config_params.max_run_steps:
			  (article_batch, abstract_batch, targets, article_lens, abstract_lens,
			   loss_weights, _, _) = data_batcher.NextBatch()
			  (_, summaries, loss, train_step) = model.run_train_step(
			      sess, article_batch, abstract_batch, targets, article_lens,
			      abstract_lens, loss_weights)

			  summary_writer.add_summary(summaries, train_step)
			  running_avg_loss = self._RunningAvgLoss(
			      running_avg_loss, loss, summary_writer, train_step)
			  step += 1
			  if step % 100 == 0:
			    summary_writer.flush()
			sv.Stop()
			return running_avg_loss	

	def eval(self,model, data_batcher):
		"""Runs model eval."""
		data_generator = data_batcher._data_generator
		model.build_graph()
		saver = tf.train.Saver()
		summary_writer = tf.train.SummaryWriter(self._config_params.eval_dir)
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		running_avg_loss = 0
		step = 0
		while True:
			time.sleep(self._config_params.eval_interval_secs)
			try:
			  	ckpt_state = tf.train.get_checkpoint_state(self._config_params.log_root)
			except tf.errors.OutOfRangeError as e:
			  	tf.logging.error('Cannot restore checkpoint: %s', e)
			  	continue
			if not (ckpt_state and ckpt_state.model_checkpoint_path):
			  	tf.logging.info('No model to eval yet at %s', self._config_params.train_dir)
			  	continue

			tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
			saver.restore(sess, ckpt_state.model_checkpoint_path)

			(article_batch, abstract_batch, targets, article_lens, abstract_lens,
			 loss_weights, _, _) = data_batcher.NextBatch()
			(summaries, loss, train_step) = model.run_eval_step(
			    sess, article_batch, abstract_batch, targets, article_lens,
			    abstract_lens, loss_weights)
			tf.logging.info(
			    'article:  %s',
			    ' '.join(data_generator.Ids2Words(article_batch[0][:].tolist())))
			tf.logging.info(
			    'abstract: %s',
			    ' '.join(data_generator.Ids2Words(abstract_batch[0][:].tolist())))

			summary_writer.add_summary(summaries, train_step)
			running_avg_loss = self._RunningAvgLoss(
			    running_avg_loss, loss, summary_writer, train_step)
			if step % 100 == 0:
			 	summary_writer.flush()

	def decode(self,model,batcher):
		hps = batcher._hps
		vocab = batcher._vocab
		config_params = self._config_params
		decoder = BSDecoder(model, batcher, hps, vocab,config_params)
		decoder.DecodeLoop()	


	def _RunningAvgLoss(self,loss, running_avg_loss, summary_writer, step, decay=0.999):
		"""Calculate the running average of losses."""
		if running_avg_loss == 0:
			running_avg_loss = loss
		else:
			running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
		running_avg_loss = min(running_avg_loss, 12)
		loss_sum = tf.Summary()
		loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
		summary_writer.add_summary(loss_sum, step)
		sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
		return running_avg_loss
