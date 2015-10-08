import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
#from sklearn.manifold import TSNE
from tsne import tsne


def train(X, Y):
    X = X.as_matrix()
    Y = Y.as_matrix()

    dims = X.shape[1]
    nb_classes = 1

    model = Sequential()
    model.add(Dense(dims, 256))
    model.add(Activation('tanh'))
    model.add(Dense(256, 256))
    model.add(Activation('tanh'))
    #model.add(Dense(128, 128))
    #model.add(Activation('tanh'))
    model.add(Dense(256, 256))
    model.add(Activation('tanh'))
    model.add(Dense(256, 1))
    
    optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    model.compile(loss='mean_absolute_error', optimizer=optimizer)

    print("Training model...")

    model.fit(X, Y, nb_epoch=30, batch_size=32, validation_split=0.15, show_accuracy=True)

    score = model.evaluate(X, Y, batch_size=32)
    print score
    return model, score

def test(X):
    import pdb; pdb.set_trace()

def main():
    training_input_file = 'data/training_vector.csv'
    test_input_file = 'data/test_vector.csv'

    training_data = pd.read_csv(training_input_file)
    test_data = pd.read_csv(test_input_file)

    model, score = train(training_data.drop(['Sales'], axis=1), training_data['Sales'])
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()