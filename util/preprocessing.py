import os

import pm4py
import tensorflow as tf
import numpy as np
from tqdm import tqdm


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

    # extract the traces from the data
    print('Extracting traces...')
    # expecting the case name in 'case:concept:name' and the activity name in 'concept:name'
    for case in tqdm(df['case:concept:name'].unique()):
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


def traces_to_embedding_indices(traces, vocab):
    # convert the traces to embedding indices of the corresponding activities
    vect_traces = []

    print('Converting traces to embedding indices...')
    for trace in tqdm(traces):
        vect_traces.append([vocab[w] for w in trace])

    return vect_traces


def traces_to_one_hot(traces, vocab):
    # Convert the list of traces to one-hot encoded vectors
    encoded_traces = []
    print('Converting traces to one-hot-encoding...')
    for trace in tqdm(traces):
        indices = [vocab[act] for act in trace]
        trace_oh = tf.one_hot(indices, depth=len(vocab))
        encoded_traces.append(trace_oh)

    return encoded_traces


def compute_sequences(vect_traces, window_length):
    # Create input-output pairs
    X = []
    y = []
    print('Generating sequence-target pairs from traces for next-activity-prediction...')
    for tv in tqdm(vect_traces):
        for i in range(len(tv) - window_length):
            X.append(tv[i:i + window_length])
            y.append(tv[i + window_length])
    return np.array(X), np.array(y)


def preprocess_dataset(dataset_name, print_summary=True, summary_to_file=False, save_to_file=True):
    # read xes data
    print('Reading xes data...')
    df = xes_to_csv('data/event_logs/{0}.xes'.format(dataset_name))

    traces = extract_traces(df)

    if print_summary or summary_to_file:
        num_events = len(df)
        num_activities = len(df["concept:name"].unique())
        num_traces = len(traces)
        mean_trace_len = np.sum([len(t) for t in traces]) / num_traces
        std_trace_len = np.std([len(t) for t in traces])
        min_trace_len = min([len(t) for t in traces])
        max_trace_len = max([len(t) for t in traces])


        if print_summary:
            print('*' * 120)
            print('Dataset Summary {0}:'.format(dataset_name))
            print('*' * 120)
            print('Number of traces: {0}'.format(num_traces))
            print('Mean trace length: {:.2f}'.format(mean_trace_len))
            print('Standard deviation trace length: {0:.2f}'.format(std_trace_len))
            print('Minimum trace length: {0}'.format(min_trace_len))
            print('Maximum trace length: {0}'.format(max_trace_len))
            print('Number of (unique) activities: {0}'.format(num_activities))
            print('Number of total events: {0}'.format(num_events))
            print('*' * 120)

        if summary_to_file:
            with open('./output/preprocessing/dataset_summary_{0}.txt'.format(dataset_name), 'w') as f:
                f.write('*' * 120 + '\n')
                f.write('Dataset Summary {0}:\n'.format(dataset_name))
                f.write('*' * 120 + '\n')
                f.write('Number of traces: {0}\n'.format(num_traces))
                f.write('Average trace length: {:.2f}\n'.format(mean_trace_len))
                f.write('Standard deviation trace length: {0:.2f}\n'.format(std_trace_len))
                f.write('Minimum trace length: {0}\n'.format(min_trace_len))
                f.write('Maximum trace length: {0}\n'.format(max_trace_len))
                f.write('Number of (unique) activities: {0}\n'.format(num_activities))
                f.write('Number of total events: {0}\n'.format(num_events))

    if save_to_file:
        # Create the directory if it does not exist already
        preprocessed_dir = './data/preprocessed/'
        os.makedirs(preprocessed_dir, exist_ok=True)
        np.save(preprocessed_dir + 'dataset_{0}_traces.npy'.format(dataset_name), traces)
    return traces


def prepare_data(padded_traces, act2vec_dict, window_sizes, dataset_name='', print_summary=True, summary_to_file=True):
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
            traces_wv = traces_to_embedding_indices(padded_traces[window_size], vocab)
            X_wv, y_wv = compute_sequences(traces_wv, window_size)
            # one hot encoding for labels
            y_wv = tf.one_hot(y_wv, depth=len(vocab)).numpy()

            data['WV'][(technique_name, window_size)] = {
                'X': X_wv,
                'y': y_wv
            }

    # one-hot encoding
    vocab_oh = get_vocabulary(padded_traces[window_sizes[0]])
    for window_size in window_sizes:
        traces_oh = traces_to_one_hot(padded_traces[window_size], vocab_oh)
        X_oh, y_oh = compute_sequences(traces_oh, window_size)

        data['OH'][window_size] = {
            'X': X_oh,
            'y': y_oh
        }

    if print_summary:
        print('*' * 120)
        print('Input Preparation Summary {0}:'.format(dataset_name))
        print('*' * 120)
        for window_size in window_sizes:
            # does not matter since all are the same for same window size
            X_oh, y_oh = data['OH'][window_size]['X'], data['OH'][window_size]['y']
            print('window_size {0}: X-shape: {1}, y-shape: {2}'.format(window_size, X_oh.shape, y_oh.shape))
        print('*' * 120)

    if summary_to_file:
        # Create the directory if it does not exist already
        os.makedirs('./output/preprocessing', exist_ok=True)

        with open('./output/preprocessing/input_preparation_summary_{0}.txt'.format(dataset_name), 'w') as f:
            f.write('*' * 120 + '\n')
            f.write('Data Preparation Summary {0}:\n'.format(dataset_name))
            f.write('*' * 120 + '\n')
            for window_size in window_sizes:
                # does not matter since all are the same for same window size
                X_oh, y_oh = data['OH'][window_size]['X'], data['OH'][window_size]['y']
                f.write('OH, window_size {0}: X-shape: {1}, y-shape: {2}\n'.format(window_size, X_oh.shape, y_oh.shape))
            f.write('*' * 120)

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
