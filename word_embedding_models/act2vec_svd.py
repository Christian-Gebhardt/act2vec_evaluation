import numpy as np
from sklearn.preprocessing import MinMaxScaler

from util.preprocessing import build_co_occurrence_matrix, calculate_ppmi_matrix

"""
Compute activity embeddings using Singular Value Decomposition (SVD) on the PPMI matrix, similar to Levy et al. 
(https://papers.nips.cc/paper_files/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html)
"""


def act2vec_svd(traces, embedding_dim, window_size):
    # build co-occurrence matrix
    coc_matrix, vocab = build_co_occurrence_matrix(traces, window_size)

    # calculate PPMI matrix
    ppmi_matrix = calculate_ppmi_matrix(coc_matrix, vocab)

    if embedding_dim > len(vocab):
        raise ValueError("Embedding dimension cannot be greater than vocabulary size.")

    # perform svd on co-occurence matrix
    U, sigma, VT = np.linalg.svd(ppmi_matrix)

    # truncate SVD components into word vectors
    embeddings = U[:, :embedding_dim]

    scaler = MinMaxScaler(feature_range=(-1, 1))

    # fit and transform the embeddings using MinMaxScaler to interval -1 and 1
    normalized_embeddings = scaler.fit_transform(embeddings)

    return normalized_embeddings, vocab
