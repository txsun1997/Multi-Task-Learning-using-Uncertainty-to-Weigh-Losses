import os
import logging
import numpy as np


def load_word_emb(path, embed_dim, vocab, save_path=None):
    ''' load word embedding from file '''

    logger = logging.getLogger('train.load_word_emb')
    if save_path is None:
        save_path = 'data/word_embedding.npy'
    if os.path.exists(save_path):
        logger.info('Loading existed word embeddings from {}.'.format(save_path))
        word_embedding = np.load(save_path)
        logger.info('Word embedding finished.')
        return word_embedding

    word_embedding = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim))
    word_embedding[0] = np.zeros((1, embed_dim))

    OOV = set()
    with open(path, encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            values = line.split()
            word = values[0]
            idx = vocab.to_index(word)
            if idx == 1:
                OOV.add(word)
                continue
            else:
                for i in range(embed_dim):
                    word_embedding[idx][i] = float(values[i + 1])

    logger.info('Pre-trained word embeddings loaded. OOV: {}'.format(len(OOV)))
    logger.info('Dumping pre-trained word embeddings in {}.'.format(save_path))
    np.save(save_path, word_embedding)
    logger.info('Word embedding finished.')
    return word_embedding
