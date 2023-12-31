import datetime
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm

from util.model_builder import build_model
from scipy import stats


def evaluate(model_names, dataset_name, model_params_list, training_params_list, data, kfold_opt=None,
             print_to_console=True, save_to_file=False, save_to_file_pred=False):
    # prepare k-fold cross validation (5x2 CV for McNemars Test)
    if kfold_opt is None:
        kfold_opt = {'n_splits': 2, 'n_repeats': 5, 'random_state': 729504}

    # use the same k-fold split for every model for fair comparison
    kf = RepeatedKFold(**kfold_opt)

    # create progress bar
    # Calculate total number of (model, training_param) pairs

    len_oh = len([model_name for model_name in model_names if 'OH' in model_name]) * len(training_params_list) \
             * len(model_params_list['OH'])

    num_techniques = len(model_params_list['WV'].keys())
    params_lists = model_params_list['WV'].values()

    len_wv = len([model_name for model_name in model_names if 'WV' in model_name]) * len(training_params_list) * \
             num_techniques * sum(len(params_list) for params_list in params_lists)

    total_pairs = len_oh + len_wv

    # evaluate the models
    print('Evaluating models...')
    # Create a single progress bar
    progress_bar = tqdm(total=total_pairs, desc='Progress')

    # loop over all selected models
    for model_name in model_names:
        # evaluate model for all specified training params
        for training_params in training_params_list:

            # models with WV input
            if model_name in ['LSTM_WV', 'FNN_WV']:
                for act2vec_technique, params_list in model_params_list['WV'].items():
                    for params in params_list:
                        key = (act2vec_technique, params['window_size'])
                        X_wv = data['WV'][key]['X']
                        y_wv = data['WV'][key]['y']
                        evaluate_model(model_name, dataset_name, params, X_wv, y_wv, kf,
                                       training_params=training_params,
                                       act2vec_technique_name=act2vec_technique, print_to_console=print_to_console,
                                       save_to_file=save_to_file, save_to_file_pred=save_to_file_pred)
                        # Update the progress bar
                        progress_bar.update(1)
            # models with OH input
            elif model_name in ['LSTM_OH', 'FNN_OH']:
                for params in model_params_list['OH']:
                    key = params['window_size']
                    X_oh = data['OH'][key]['X']
                    y_oh = data['OH'][key]['y']
                    evaluate_model(model_name, dataset_name, params, X_oh, y_oh, kf, training_params=training_params,
                                   print_to_console=print_to_console, save_to_file=save_to_file)
                    # Update the progress bar
                    progress_bar.update(1)
            else:
                raise ValueError('Model name {0} not found.'.format(model_name))
    # Close the progress bar
    progress_bar.close()


def evaluate_model(model_name, dataset_name, model_params, X, y, kfold, training_params=None, act2vec_technique_name='',
                   print_to_console=True,
                   save_to_file=False, save_to_file_pred=False):
    results = []
    training_times = []

    # initialize a dictionary to store predictions for all k-folds
    kfold_predictions = {}

    # training history
    training_history = []

    # iterate over the generated k_folds.
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
        history_callback = model.fit(X_train, y_train, **training_params)
        end_time = time.time()

        elapsed_time = end_time - start_time
        training_times.append(elapsed_time)

        training_history.append(history_callback.history)

        # evaluate the model on the test data
        results.append(model.evaluate(X_test, y_test, verbose=0))

        if save_to_file_pred:
            # Predict probabilities and labels for the test data
            predicted_probabilities = model.predict(X_test, verbose=0)
            predicted_labels = np.argmax(predicted_probabilities, axis=-1)

            # Store data for this k-fold in the dictionary
            kfold_predictions[f'kfold_{i + 1}'] = {
                "predicted_probabilities": predicted_probabilities,
                "predicted_labels": predicted_labels,
                "true_labels": y_test
            }

    # Get the current date and time
    current_time = datetime.datetime.now()

    # Format the current date and time as "dd-mm-yy_hh:mm"
    formatted_time = current_time.strftime("%d-%m-%y_%H:%M")

    header = {
        'model_name': model_name,
        'dataset': dataset_name.upper(),
        'time': formatted_time,
        'epochs': training_params['epochs'],
        'batch_size': training_params['batch_size'],
        'window_size': model_params['window_size']
    }

    if 'WV' in model_name:
        header['technique'] = act2vec_technique_name
        header['embedding_dim'] = model_params['embedding_dim']

    # print the results to console
    if print_to_console:
        print_results(model_name, header, results, training_times)

    # save the results as csv (pandas df)

    if save_to_file:
        results_dir = './output/evaluation/'
        pred_dir = './output/prediction/'
        training_history_dir = './output/training_history/'

        # Create the directories if it does not exist already
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(training_history_dir, exist_ok=True)

        setup_name = '{0}_{1}_{2}_EP{3}B{4}_WIN{5}'.format(
            model_name, dataset_name.upper(), formatted_time, training_params['epochs'],
            training_params['batch_size'], model_params['window_size'])

        if 'WV' in model_name:
            if act2vec_technique_name:
                setup_name += "_{0}".format(act2vec_technique_name)
            setup_name += "_DIM{0}".format(model_params['embedding_dim'])

        # save the k-fold predictions to a .npy file

        if save_to_file_pred:
            filename_pred = os.path.join(pred_dir, setup_name) + '_PRED.npy'
            np.save(filename_pred, kfold_predictions)

        # save the k-fold training history to a .npy file
        filename_history = os.path.join(training_history_dir, setup_name + '_HISTORY.npy')
        np.save(filename_history, training_history)

        # save the results to .csv file
        filename_csv = os.path.join(results_dir, setup_name)
        save_results_to_csv(model_name, results, training_times, filename_csv)


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


def print_results(model_name, header, results, training_times):
    sum_metrics = [0] * (len(results[0]) - 1)  # Initialize a list to store the sum of metrics
    print('*' * 120)
    for i in range(len(results)):
        print("k-fold {}: accuracy: {:.4f}, precision: {:.4f}, recall: "
              "{:.4f}, top-3-accuracy: {:.4f}, training_time: {:.4f}".format(
            i + 1, results[i][1], results[i][2], results[i][3], results[i][4], training_times[i]))

        for j in range(1, len(results[i])):
            sum_metrics[j - 1] += results[i][j]

    average_metrics = [sum_metric / len(results) for sum_metric in sum_metrics]

    setup = "{}, {}, EP{}, B{}, W{}".format(model_name, header['dataset'], header['epochs'],
                                            header['batch_size'], header['window_size'])
    if model_name in ['LSTM_WV', 'FNN_WV']:
        setup += ", {}, DIM{}".format(header['technique'], header['embedding_dim'])

    print('*' * 120)
    print("SUMMARY {} => average accuracy: {:.4f}, average precision: {:.4f}, average recall: {:.4f}, "
          "average top-3-accuracy: {:.4f}"
          .format(setup, average_metrics[0], average_metrics[1], average_metrics[2], average_metrics[3]))


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
            sum_metrics[j] += results[i][j + 1]

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
    df.to_csv(filename + '.csv', index=False)


# calculate paired students t-test for hypotheses testing
def paired_students_ttest(kfold_results1, kfold_results2, alpha):
    num_results = min(kfold_results1, kfold_results2)

    res_differences = []
    # calculate metric differences for every fold
    for i in range(num_results):
        diff = kfold_results1[i] - kfold_results2[i]
        res_differences.append(diff)

    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(res_differences, np.zeros(len(res_differences)))

    # if p_value is smaller than alpha reject h0
    reject_null_hypothesis = True if p_value < alpha else False

    return t_statistic, p_value, reject_null_hypothesis
