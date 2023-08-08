import os

import numpy as np

from eval.evaluation import evaluate
from util import prepare_evaluation
from util.prepare_evaluation import prepare_data_and_params
from util.preprocessing import prepare_data, add_padding, preprocess_dataset
from word_embedding_models.act2vec_sgns import act2vec_sgns

if __name__ == '__main__':

    dataset_name = 'helpdesk'

    traces = []

    # load preprocessed traces if npy file exists
    if os.path.exists('./data/preprocessed/dataset_{0}_traces.npy'.format(dataset_name)):
        traces = np.load('./data/preprocessed/dataset_{0}_traces.npy'.format(dataset_name), allow_pickle=True)
    else:
        traces = preprocess_dataset(dataset_name)

    # define hyperparams for embeddings
    window_sizes = [5]
    embedding_dims = [128]
    act2vec_techniques = ['SGNS']

    # define training params
    epoch_nums = [3]
    batch_sizes = [64]

    evaluation_data, model_params, training_params = prepare_data_and_params(
        act2vec_techniques, traces, dataset_name, embedding_dims, window_sizes,
        epoch_nums, batch_sizes)

    # define models to evaluate and their hyperparams
    model_names = ['FNN_WV', 'FNN_OH', 'LSTM_WV', 'LSTM_OH']

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

    # evaluate the models
    print('Evaluating models...')
    evaluate(model_names, model_params, training_params, evaluation_data, save_to_file=True)
