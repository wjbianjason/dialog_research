from __future__ import print_function

import os

from keras.models import model_from_json
from keras.utils.data_utils import get_file



class SequenceModel:
    """
    The offline class to train and test model for sequence to sequence learning.
    """

    def __init__(self, input_length=None, model=None, model_type=None):
        if model:
            self.model = model
        else:
            print("no model input")
        print(self.model.summary())

    def predict(self, x_in):
        return self.model.predict(x_in)

    def predict_class(self,x_in):
        return self.model.predict_classes(x_in)

    def train(self, x, y, epoch=1):
        print('Training with input shape', x.shape, ' and output shape ', y.shape)
        self.model.fit(x, y, nb_epoch=epoch)

    def train_generator(self,input_gen,samples_per_epoch=10000,epoch=10,valid_data=None):
        self.model.fit_generator(input_gen,samples_per_epoch=samples_per_epoch,nb_epoch=epoch,validation_data=valid_data,verbose=1)


    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def save(self, file_name):
        json_string = self.model.to_json()
        json_string = json_string.replace('SimpleSeq2seq', 'Sequential')
        json_file_name, h5_file_name = SequenceModel.get_full_file_names(file_name)
        open("./model_bak/"+json_file_name, 'w').write(json_string)
        self.model.save_weights("./model_bak/"+h5_file_name, overwrite=True)
        print('Saved model to file ', file_name)


    @staticmethod
    def load(file_name):
        json_file_name, h5_file_name = SequenceModel.get_full_file_names(file_name)
        model = model_from_json(open("./model_bak/"+json_file_name, 'r').read())
        model.compile(optimizer='rmsprop', loss='mse')
        model.load_weights("./model_bak/"+h5_file_name)
        print('Loaded file ', file_name)
        return SequenceModel(model=model)

    @staticmethod
    def delete_model(file_name):
        for full_file_name in SequenceModel.get_full_file_names(file_name):
            try:
                os.remove(full_file_name)
            except OSError:
                pass

    @staticmethod
    def get_full_file_names(file_name):
        return file_name + '.json', file_name + '.h5'

    @staticmethod
    def isfile(model_file_name):
        json_file_name, h5_file_name = SequenceModel.get_full_file_names(model_file_name)
        return os.path.isfile(json_file_name) and os.path.isfile(h5_file_name)
