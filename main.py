import os

import numpy as np

from eval.evaluation import evaluate
from util.prepare_evaluation import prepare_data_and_params
from util.preprocessing import preprocess_dataset

from tensorflow.keras.callbacks import EarlyStopping


if __name__ == '__main__':

    dataset_name = 'helpdesk'

    traces = []

    # load preprocessed traces if npy file exists
    if os.path.exists('./data/preprocessed/dataset_{0}_traces.npy'.format(dataset_name)):
        traces = np.load('./data/preprocessed/dataset_{0}_traces.npy'.format(dataset_name), allow_pickle=True)
    else:
        traces = preprocess_dataset(dataset_name, summary_to_file=True, save_to_file=False)

    # define hyperparams for embeddings
    window_sizes = [10]
    embedding_dims = [8, 32, 128, 256]
    act2vec_techniques = ['CBOW', 'SGNS', 'RANDOM']

    # define training params
    epoch_nums = [500]
    batch_sizes = [128]

    # define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    validation_split = 0.2

    # define manually here instead of prepare_data_and_params (due to additional params with early stopping)
    training_params = [{
        'epochs': epoch_nums[0],
        'batch_size': batch_sizes[0],
        'validation_split': validation_split,
        'callbacks': [early_stopping],
        'verbose': 0
    }]

    evaluation_data, model_params, _ = prepare_data_and_params(
        act2vec_techniques, traces, dataset_name, embedding_dims, window_sizes,
        epoch_nums, batch_sizes)

    # define models to evaluate and their hyperparams
    model_names = ['FNN_WV', 'LSTM_WV']

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
    evaluate(model_names, dataset_name, model_params, training_params, evaluation_data, save_to_file=True)
