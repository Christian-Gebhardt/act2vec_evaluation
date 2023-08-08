import numpy as np

from util.preprocessing import get_vocabulary


# initialize random activity embeddings (range -1 to 1) as baseline
def act2vec_random(traces, embedding_dim):
    vocab = get_vocabulary(traces)
    vocab_size = len(vocab)

    act_embeddings = np.random.uniform(low=-1.0, high=1.0, size=(vocab_size, embedding_dim))

    return act_embeddings, vocab
