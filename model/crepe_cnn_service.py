

from mms.model_service.mxnet_model_service import MXNetBaseService
from mxnet.gluon.data import ArrayDataset
import numpy as np
import mxnet as mx
from mxnet import nd

ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
ALPHABET_INDEX = {letter: index for index, letter in enumerate(ALPHABET)} # { a: 0, b: 1, etc}
FEATURE_LEN = 1014 # max-length in characters for one document

# Encode the text for processing
def encode(text):
    encoded = np.zeros([len(ALPHABET), FEATURE_LEN], dtype='float32')
    review = text.lower()[::-1]
    i = 0
    for letter in text:
        if i >= FEATURE_LEN:
            break;
        if letter in ALPHABET_INDEX:
            encoded[ALPHABET_INDEX[letter]][i] = 1
        i += 1
    return encoded

class TextClassicationService(MXNetBaseService):
    
    def _preprocess(self, data):
        assert (len(data[0]) <= 1), 'Too many inputs {}'.format(len(data))
        encoded_data = [encode(d) for d in data[0]]
        return list(map(mx.nd.array, [encoded_data]))
    
    def _postprocess(self, data):
        data = data[0]
        softmax = nd.exp(data) / nd.sum(nd.exp(data))[0]
        values = {val: float(int(softmax[0][i].asnumpy()*1000)/1000.0) for i, val in enumerate(self.labels)}
        index = int(nd.argmax(data, axis=1).asnumpy()[0])
        predicted = self.labels[index]
        return {'predicted':predicted, 'confidence':values}