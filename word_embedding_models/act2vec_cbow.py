from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler

"""
compute activity embeddings using Continuous Bag of Words (CBOW) model, introduced in Mikolov et. al
(https://doi.org/10.48550/arXiv.1301.3781), implemented in the GenSim library.
"""


def act2vec_cbow(traces, embedding_dim, window_size):
    cbow_model = Word2Vec(sentences=traces, vector_size=embedding_dim, window=window_size, min_count=1, workers=4)

    embeddings = cbow_model.wv.vectors
    vocab = cbow_model.wv.key_to_index

    scaler = MinMaxScaler(feature_range=(-1, 1))

    # fit and transform the embeddings using MinMaxScaler to interval -1 and 1
    normalized_embeddings = scaler.fit_transform(embeddings)

    return normalized_embeddings, vocab
