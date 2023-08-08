from gensim.models import Word2Vec

"""
compute activity embeddings using Skip-Gram model with Negative Sampling (SGNS), introduced in Mikolov et. al
(https://doi.org/10.48550/arXiv.1310.4546), implemented in the GenSim library.
"""
def act2vec_sgns(traces, embedding_dim, window_size):
    sgns_model = Word2Vec(sentences=traces, vector_size=embedding_dim, window=window_size, min_count=1, workers=4, sg=1)
    return sgns_model.wv.vectors, sgns_model.wv.key_to_index
