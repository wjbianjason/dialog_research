from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from seq2seq.models import SimpleSeq2seq,AttentionSeq2seq
from keras.layers import Embedding

class ModelFactory(object):

    @staticmethod
    def get_single_layer_LSTM(input_shape, nb_classes, output_length, hidden_units):
        if not output_length:
            raise Exception('Output Length required for sequence model')
        word2vec_dimension = input_shape[1]
        model = AttentionSeq2seq(input_dim=word2vec_dimension, input_length = 6, depth = 2 ,hidden_dim=hidden_units, output_length=output_length,
                              output_dim=nb_classes)
        print(nb_classes)
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model

    @staticmethod
    def get_first_lstm_model(input_shape, nb_classes):
        print('Getting First LSTM model of shape ', input_shape, ' output classes ', nb_classes)
        model = Sequential()
        model.add(LSTM(128, input_shape=input_shape))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    @staticmethod
    def get_simplest_model():
        model = Sequential()
        model.add(Dense(1, input_dim=1))
        return model

    @staticmethod
    def get_2layer_1k_model(input_shape, nb_classes, output_length):
        if not output_length:
            raise Exception('Output Length required for sequence model')
        word2vec_dimension = input_shape[1]
        model = SimpleSeq2seq(input_dim=word2vec_dimension, hidden_dim=1000, output_length=output_length, depth=2,
                              output_dim=nb_classes)
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model
    
    @staticmethod
    def get_simple_model(model_param,vovab_in_size,vocab_out_size,emb_dims):
        model = Sequential()
        model.add(Embedding(vovab_in_size,emb_dims,input_length=model_param.enc_timesteps))
        model.add(SimpleSeq2seq(input_dim=emb_dims,hidden_dim=1000,output_length=model_param.dec_timesteps,depth=2,
            output_dim=vocab_out_size))
        model.compile(loss="categorical_crossentropy",optimizer='rmsprop')
        return model

    @staticmethod
    def get_attention_model(model_param,vovab_in_size,vocab_out_size,emb_dims):
        model = Sequential()
        model.add(Embedding(vovab_in_size,emb_dims,input_length=model_param.enc_timesteps))
        model.add(AttentionSeq2seq(input_length=model_param.enc_timesteps,input_dim=emb_dims,hidden_dim=200,output_length=model_param.dec_timesteps,depth=1,
            output_dim=vocab_out_size))
        model.compile(loss="categorical_crossentropy",optimizer='rmsprop')
        return model