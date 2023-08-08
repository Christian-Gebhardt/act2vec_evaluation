import datetime
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from util.model_builder import build_model


def evaluate(model_names, model_params_list, training_params_list, data, kfold_opt=None,
             print_to_console=True, save_to_file=True):
    # prepare k-fold cross validation
    if kfold_opt is None:
        kfold_opt = {'n_splits': 5, 'shuffle': True}

    # use the same k-fold split for every model for fair comparison
    kf = KFold(**kfold_opt)

    # loop over all selected models
    for model_name in model_names:
        # evaluate model for all specified training params
        for training_params in training_params_list:

            # models with WV input
            if model_name in ['LSTM_WV', 'FNN_WV']:
                for act2vec_technique, params in model_params_list['WV']:
                    key = (act2vec_technique, params['window_size'])
                    X_wv = data['WV'][key]['X']
                    y_wv = data['WV'][key]['y']
                    evaluate_model(model_name, params, X_wv, y_wv, kf, training_params=training_params,
                                   act2vec_technique_name=act2vec_technique, print_to_console=print_to_console,
                                   save_to_file=save_to_file)
            # models with OH input
            elif model_name in ['LSTM_OH', 'FNN_OH']:
                for params in model_params_list['OH']:
                    key = params['window_size']
                    X_oh = data['OH'][key]['X']
                    y_oh = data['OH'][key]['y']
                    evaluate_model(model_name, params, X_oh, y_oh, kf, training_params=training_params,
                                   print_to_console=print_to_console, save_to_file=save_to_file)
            else:
                raise ValueError('Model name {0} not found.'.format(model_name))


def evaluate_model(model_name, model_params, X, y, kfold, training_params=None, act2vec_technique_name='',
                   print_to_console=True,
                   save_to_file=True):
    results = []
    training_times = []

    # Iterate over the generated k_folds.
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        model = build_model(model_name, model_params)

        if training_params is None:
            # set default training params
            training_params = {
                'epochs': 10,
                'batch_size': 32,
                'verbose': 0
            }

        # Measuring training time
        start_time = time.time()
        # Train your model
        model.fit(X_train, y_train, **training_params)
        end_time = time.time()

        elapsed_time = end_time - start_time
        training_times.append(elapsed_time)

        # evaluate the model on the test data
        results.append(model.evaluate(X_test, y_test, verbose=0))

    # print the results to console
    if print_to_console:
        print_results(model_name, results, training_times)

    # save the results as csv (pandas df)

    if save_to_file:
        # Get the current date and time
        current_time = datetime.datetime.now()

        # Format the current date and time as "dd-mm-yy_hh:mm"
        formatted_time = current_time.strftime("%d-%m-%y_%H:%M")

        results_dir = "./res_{0}".format(formatted_time)

        # Create the directory if they do not exist already
        os.makedirs(results_dir, exist_ok=True)

        filename = os.path.join(results_dir, "{0}_EPOCHS{1}_WINDOW{2}".format(model_name, training_params['epochs'],
                                                                              model_params['window_length']))

        if 'WV' in model_name:
            if act2vec_technique_name:
                filename += "_{0}".format(act2vec_technique_name)
            filename += "_DIM{0}".format(model_params['embedding_dim'])

        save_results_to_csv(model_name, results, training_times, filename)


def plot_embeddings_tsne(act_vectors, vocab, tsne_options=None, title=''):
    if tsne_options is None:
        tsne_options = {'n_components': 3, 'random_state': 42}

    tsne_model = TSNE(**tsne_options)

    embeddings_tsne = np.array(tsne_model.fit_transform(act_vectors))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], embeddings_tsne[:, 2])

    for i, word in enumerate(vocab):
        ax.text(embeddings_tsne[i, 0], embeddings_tsne[i, 1], embeddings_tsne[i, 2], word, fontsize=5)

    plt.title(title)

    plt.show()


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0.0
    else:
        return dot_product / (norm_vector1 * norm_vector2)


# compute similarity matrix of the vectors where matrix[i, j] is the cosine similarity of words i and j
def compute_similarity_matrix(vectors):
    num_vectors = len(vectors)
    similarity_matrix = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(i, num_vectors):
            similarity = cosine_similarity(vectors[i], vectors[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    return similarity_matrix


def get_top_similarities(similarity_matrix, target_vector_index, n):
    # find top n similar vectors to a target vector
    target_vector_similarities = similarity_matrix[target_vector_index]
    sorted_indices = np.argsort(target_vector_similarities)[::-1]

    top_n_indices = sorted_indices[1:n + 1]  # Exclude the target vector itself
    top_n_similarities = target_vector_similarities[top_n_indices]

    # return dict with indices as key and similarities as values
    return zip(top_n_indices, top_n_similarities)


def print_results(model_name, results, training_times):
    sum_metrics = [0] * (len(results[0]) - 1)  # Initialize a list to store the sum of metrics
    print('*' * 120)
    for i in range(len(results)):
        print("{} => k-fold {}: accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, top-3-accuracy: {:.4f} "
              "training_time: {:.4f}".format(model_name, i + 1, results[i][1], results[i][2], results[i][3],
                                             results[i][4], training_times[i]))
        for j in range(1, len(results[i])):
            sum_metrics[j - 1] += results[i][j]

    average_metrics = [sum_metric / len(results) for sum_metric in sum_metrics]

    print('*' * 120)
    print("SUMMARY {} => average accuracy: {:.4f}, average precision: {:.4f}, average recall: {:.4f}, "
          "average top-3-accuracy: {:.4f}"
          .format(model_name, average_metrics[0], average_metrics[1], average_metrics[2], average_metrics[3]))


def save_results_to_csv(model_name, results, training_times, filename):
    sum_metrics = [0] * (len(results[0]) - 1)  # Initialize a list to store the sum of metrics
    sum_training_time = 0

    # Create a list to store rows (as dictionaries) for the DataFrame
    rows = []

    for i in range(len(results)):
        k_fold = i + 1
        accuracy, precision, recall, top_3_accuracy, training_time = \
            results[i][1], results[i][2], results[i][3], results[i][4], training_times[i]

        # Append the row to the list of rows
        rows.append({"model_name": model_name,
                     "k-fold": k_fold,
                     "accuracy": accuracy,
                     "precision": precision,
                     "recall": recall,
                     "top-3-accuracy": top_3_accuracy,
                     "training_time": training_time})

        # Calculate the sum of metrics
        for j in range(0, len(results[i]) - 1):
            sum_metrics[j] += results[i][j]
            sum_training_time += training_times[i]

    # Calculate the average metrics
    average_metrics = [sum_metric / len(results) for sum_metric in sum_metrics]
    average_training_time = sum_training_time / len(results)

    # Append the row for average metrics to the list of rows
    rows.append({"model_name": model_name,
                 "k-fold": "Average",
                 "accuracy": average_metrics[0],
                 "precision": average_metrics[1],
                 "recall": average_metrics[2],
                 "top-3-accuracy": average_metrics[3],
                 "training_time": average_training_time})

    # Create the DataFrame from the list of rows
    df = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
