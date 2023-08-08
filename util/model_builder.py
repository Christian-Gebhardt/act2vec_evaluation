from keras.metrics import Accuracy, Precision, Recall, TopKCategoricalAccuracy

from pm_models.next_actvitiy_prediction.fnn_oh import FNN_OH
from pm_models.next_actvitiy_prediction.fnn_wv import FNN_WV
from pm_models.next_actvitiy_prediction.lstm_oh import LSTM_OH
from pm_models.next_actvitiy_prediction.lstm_wv import LSTM_WV


def build_model(model_name, model_params, compile_options=None):
    if compile_options is None:
        # set default compile options
        compile_options = {'loss': 'categorical_crossentropy',
                           'optimizer': 'adam', 'metrics': ['accuracy', Precision(), Recall(), TopKCategoricalAccuracy(3)]}

    # switch case for selected model by model_name
    if model_name == 'LSTM_WV':
        model = LSTM_WV(**model_params)
        model.compile(**compile_options)
        return model
    elif model_name == 'LSTM_OH':
        model = LSTM_OH(**model_params)
        model.compile(**compile_options)
        return model
    elif model_name == 'FNN_WV':
        model = FNN_WV(**model_params)
        model.compile(**compile_options)
        return model
    elif model_name == 'FNN_OH':
        model = FNN_OH(**model_params)
        model.compile(**compile_options)
        return model
    else:
        raise ValueError('Model "{0}" not found.'.format(model_name))
