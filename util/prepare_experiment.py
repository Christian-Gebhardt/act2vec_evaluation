from util.preprocessing import prepare_data, xes_to_csv, extract_traces
from word_embedding_models.act2vec_cbow import act2vec_cbow
from word_embedding_models.act2vec_random import act2vec_random
from word_embedding_models.act2vec_sgns import act2vec_sgns
from word_embedding_models.act2vec_svd import act2vec_svd


def prepare_data_and_params(act2vec_techniques, dataset_name, vocab_size, embedding_dims, window_sizes, epoch_nums,
                            batch_sizes, print):
    model_params = {
        'OH': [],
        'WV': {}
    }
    training_params = []

    # prepare traces

    # read xes data
    print('Reading xes data...')
    df = xes_to_csv('data/event_logs/{0}'.format(dataset_name))

    # extract the traces from the data
    print('Extracting traces...')
    traces = extract_traces(df)

    # compute all embeddings
    act2vec = compute_embeddings(act2vec_techniques, traces, embedding_dims, window_sizes)

    # compute X, y for OH and all WV techniques (different indices for different vocabs...)
    evaluation_data = prepare_data(traces, act2vec, window_sizes)

    # prepare WV params
    for act2vec_technique in act2vec_techniques:
        model_params['WV'][act2vec_technique] = []
        for embedding_dim in embedding_dims:
            for window_size in window_sizes:
                # access learned embeddings in dict
                vectors, vocab = act2vec[(act2vec_technique, embedding_dim, window_size)]

                model_params['WV'][act2vec_technique].append({
                    'layer_size': (32, 32),
                    'vocab_size': len(vocab),
                    'embedding_dim': embedding_dim,
                    'word_vectors': vectors,
                    'window_length': window_size
                })

    # prepare OH params
    for window_size in window_sizes:
        model_params['OH'].append({
            'layer_size': (32, 32),
            'vocab_size': vocab_size,
            'window_length': window_size
        })

    # prepare training params
    for epoch_num in epoch_nums:
        for batch_size in batch_sizes:
            training_params.append({
                'epochs': epoch_num,
                'batch_sizes': batch_size
            })

    return evaluation_data, model_params, training_params


def compute_embeddings(act2vec_techniques, traces, embedding_dims, window_sizes):
    act_vectors = {}
    for i, technique in enumerate(act2vec_techniques):
        for j, embedding_dim in enumerate(embedding_dims):
            for k, window_size in enumerate(window_sizes):
                if technique == 'CBOW':
                    act_vectors[(technique, embedding_dim, window_size)] = act2vec_cbow(traces, embedding_dim,
                                                                                        window_size)
                elif technique == 'SGNS':
                    act_vectors[(technique, embedding_dim, window_size)] = act2vec_sgns(traces, embedding_dim,
                                                                                        window_size)
                elif technique == 'SVD':
                    act_vectors[(technique, embedding_dim, window_size)] = act2vec_svd(traces, embedding_dim,
                                                                                       window_size)
                elif technique == 'RANDOM':
                    act_vectors[(technique, embedding_dim, window_size)] = act2vec_random(traces, embedding_dim)
                else:
                    raise ValueError('Technique "{0}" not found.'.format(technique))

    return act_vectors
