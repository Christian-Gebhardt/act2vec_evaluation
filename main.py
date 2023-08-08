from eval.evaluation import evaluate, plot_embeddings_tsne, compute_similarity_matrix, get_top_similarities
from util.compute_embeddings import compute_embeddings
from util.preprocessing import xes_to_csv, extract_traces, prepare_data, get_inv_vocabulary, add_padding
from word_embedding_models.act2vec_cbow import act2vec_cbow
from word_embedding_models.act2vec_random import act2vec_random
from word_embedding_models.act2vec_sgns import act2vec_sgns
from word_embedding_models.act2vec_svd import act2vec_svd
import random

if __name__ == '__main__':
    # read xes data
    print('Reading xes data...')
    df = xes_to_csv('data/event_logs/helpdesk.xes')

    # extract the traces from the data
    print('Extracting traces...')
    traces = extract_traces(df[:10000])

    # define hyperparams for embeddings
    window_size = 5
    embedding_dim = 128
    act2vec_techniques = ['SGNS']

    # add padding to traces (to predict activities at the beginning with index < window_size and prefix >= 1)
    traces = add_padding(traces, window_size)

    # computing activity embeddings
    print('Computing activity embeddings..')

    # access act_vectors dict with tuple (technique, embedding_dim, window_size) to get tuple (vectors, vocab)
    act_vectors, vocab = act2vec_sgns(traces, embedding_dim, window_size)

    """
    print('Plotting embeddings as TSNE and showing some similarities...')
    # plot_embeddings_tsne(act_vectors)
    
    similarity_matrix = compute_similarity_matrix(act_vectors)

    # pick a random activity...
    random_target_index = random.randint(0, len(vocab) - 1)
    # ...and show the top n most similar values (by cosine similarity)
    top_n = 5
    most_similar = get_top_similarities(similarity_matrix, random_target_index, top_n)
    inv_vocab = get_inv_vocabulary(vocab)
    
    print('Top {} similar activity vectors by cosine similarity to vector "{}":'.format(top_n,
                                                                                        inv_vocab[random_target_index]))
    for idx, similarity in most_similar:
        print(inv_vocab[idx] + ': ', similarity)
        
    """

    # prepare data for NN input
    print('Preparing data...')
    data = prepare_data(traces[:100], vocab, window_size)

    # define models to evaluate and their hyperparams
    model_names = ['FNN_WV', 'FNN_OH', 'LSTM_WV', 'LSTM_OH']

    model_params = {
        'LSTM_WV': [{'layer_size': (32, 32), 'vocab_size': len(vocab), 'embedding_dim': embedding_dim,
                     'word_vectors': act_vectors, 'window_length': window_size}],
        'LSTM_OH': [{'layer_size': (32, 32), 'vocab_size': len(vocab), 'window_length': window_size}],
        'FNN_WV': [{'layer_size': (32, 32), 'vocab_size': len(vocab), 'embedding_dim': embedding_dim,
                    'word_vectors': act_vectors, 'window_length': window_size}],
        'FNN_OH': [{'layer_size': (32, 32), 'vocab_size': len(vocab), 'window_length': window_size}]
    }
    # define training params
    training_params = [{
        'epochs': 10,
        'batch_size': 256,
        'verbose': 0
    }]

    # evaluate the models
    print('Evaluating models...')
    evaluate(model_names, model_params, training_params, data, save_to_file=False)
