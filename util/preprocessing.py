import pm4py
import tensorflow as tf
import numpy as np


def xes_to_csv(input_path, output_path=''):
    # convert event log from xes format to csv format
    log = pm4py.read_xes(input_path)
    dataframe = pm4py.convert_to_dataframe(log)
    if output_path:
        dataframe.to_csv(output_path)
    return dataframe


def extract_traces(df):
    # extract traces from dataframe
    traces = []

    # expecting the case name in 'case:concept:name' and the activity name in 'concept:name'
    for case in df['case:concept:name'].unique():
        trace = list(df[df['case:concept:name'] == case]['concept:name'])
        traces.append(trace)

    return traces


def add_padding(traces, window_size):
    """
    add w-1 padding to every trace, so the first activity to predict has
    prefix length 1 (second activity)
    """
    for trace in traces:
        padding = ['<PAD>'] * (window_size - 1)
        trace[:0] = padding

    return traces


def vectorize_traces(data, vocab):
    """
    convert the traces from string activity representations to integer representations
    (from vocabulary) as neural network model input later
    """
    vect_traces = []

    for trace in data:
        vect_traces.append([vocab[w] for w in trace])

    return vect_traces


def traces_to_embedding_indices(traces, vocab):
    # convert the traces to embedding indices of the corresponding activities
    vect_traces = []

    for trace in traces:
        vect_traces.append([vocab[w] for w in trace])

    return vect_traces


def traces_to_one_hot(traces, vocab):
    # Convert the list of traces to one-hot encoded vectors
    encoded_traces = []
    for trace in traces:
        indices = [vocab[act] for act in trace]
        trace_oh = tf.one_hot(indices, depth=len(vocab))
        encoded_traces.append(trace_oh)

    return encoded_traces


def prepare_sequences(vect_traces, window_length):
    # Create input-output pairs
    X = []
    y = []
    for tv in vect_traces:
        for i in range(len(tv) - window_length):
            X.append(tv[i:i + window_length])
            y.append(tv[i + window_length])
    return np.array(X), np.array(y)


def prepare_data(traces, act2vec_dict, window_sizes, dataset_name='', print_summary=True, save_to_file=True):
    # prepare data to be in appropriate input format for NN

    # OH: access with window_size
    # WV: access with (technique_name, window_size)-tuple
    data = {
        'OH': {},
        'WV': {}
    }

    # vector encoding for embeddings
    for key, value in act2vec_dict.items():
        # unroll (technique, embedding_dim, window_size) key-tuple
        technique_name, _, window_size = key
        # unroll (vectors, vocab) tuple value-tuple
        _, vocab = value

        for window_size in window_sizes:
            traces_wv = traces_to_embedding_indices(traces, vocab)
            X_wv, y_wv = prepare_sequences(traces_wv, window_size)
            # one hot encoding for labels
            y_wv = tf.one_hot(y_wv, depth=len(vocab)).numpy()

            data['WV'][(technique_name, window_size)] = {
                'X': X_wv,
                'y': y_wv
            }

    # one-hot encoding
    vocab_oh = get_vocabulary(traces)
    for window_size in window_sizes:
        traces_oh = traces_to_one_hot(traces, vocab_oh)
        X_oh, y_oh = prepare_sequences(traces_oh, window_size)

        data['OH'][window_size] = {
            'X': X_oh,
            'y': y_oh
        }

    if print_summary:
        print('*' * 120)
        print('Data Preparation Summary {0}:'.format(dataset_name))
        print('*' * 120)
        for window_size in window_sizes:
            # does not matter since all are the same for same window size
            X_oh, y_oh = data['OH'][window_size]['X'], data['OH'][window_size]['y']
            print('OH, window_size {0}: X-shape: {1}, y-shape: {2}'.format(window_size, X_oh.shape, y_oh.shape))

    if save_to_file:
        with open('data_preparation_summary_{0}.txt'.format(dataset_name), 'w') as f:
            f.write('*' * 120)
            f.write('\nData Preparation Summary {0}:\n'.format(dataset_name))
            f.write('*' * 120)
            for window_size in window_sizes:
                # does not matter since all are the same for same window size
                X_oh, y_oh = data['OH'][window_size]['X'], data['OH'][window_size]['y']
                f.write('OH, window_size {0}: X-shape: {1}, y-shape: {2}\n'.format(window_size, X_oh.shape, y_oh.shape))

    return data


# get (key, value)-dictionary with activity names as keys and indices as values
def get_vocabulary(traces):
    vocab, index = {}, 0
    for trace in traces:
        for act in trace:
            if act not in vocab:
                vocab[act] = index
                index += 1
    return vocab


# get (key, value)-dictionary with indices as keys and activity names as values
def get_inv_vocabulary(vocab):
    return {v: k for k, v in vocab.items()}


def build_co_occurrence_matrix(traces, window_size):
    # build vocabulary for activities (activities to indices)
    vocab = get_vocabulary(traces)
    v = len(vocab)

    # Init 2D co-occurrence matrix with zeroes
    coc_matrix = np.zeros((v, v), dtype=int)

    # Add padding tokens to each trace, to include words at the start and end
    padded_traces = [['<PAD>'] * window_size + trace + ['<PAD>'] * window_size for trace in traces]

    # Build co-occurrence matrix
    for trace in padded_traces:
        for i, act in enumerate(trace):
            for j in range(max(0, i - window_size), min(len(trace), i + window_size + 1)):
                if i != j and act != '<PAD>' and trace[j] != '<PAD>':
                    index_i = vocab[act]
                    index_j = vocab[trace[j]]
                    coc_matrix[index_i][index_j] += 1

    return coc_matrix, vocab


# calculate the Positive Point-wise Mutual Information (PPMI) matrix
def calculate_ppmi_matrix(co_occurrence_matrix, vocab):
    vocab_size = len(vocab)
    ppmi_matrix = np.zeros((vocab_size, vocab_size))

    total_obs = sum(co_occurrence_matrix)

    for i in range(vocab_size):
        for j in range(vocab_size):
            co_occurrences = co_occurrence_matrix[i, j]

            if co_occurrences == 0:
                continue

            # Calculate PMI
            pmi = np.log(co_occurrences * total_obs / (np.sum(co_occurrence_matrix[i, :]) *
                                                       np.sum(co_occurrence_matrix[:, j])))

            # Take only positive values for PPMI
            ppmi = max(0, pmi)

            ppmi_matrix[i, j] = ppmi

    return ppmi_matrix
