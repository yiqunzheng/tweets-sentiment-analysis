import cPickle as pickle

def load_model(model_filename='model.pkl'):
    with open(model_filename) as f:
        model = pickle.load(f)       
    return model