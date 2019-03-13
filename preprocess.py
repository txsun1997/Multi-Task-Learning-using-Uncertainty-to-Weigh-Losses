import sys
import os
import argparse
import torch
import time
import logging
from vocabulary import Vocabulary

logger = logging.getLogger('preprocess')
logger.setLevel(level=logging.DEBUG)

# Stream Handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

# File Handler
file_handler = logging.FileHandler('logs/preprocess.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(fmt='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
                                   datefmt='%Y/%m/%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


def read_instances_from_file(files, max_len, keep_case):
    ''' Collect instances and construct vocab '''

    vocab = Vocabulary()
    pos_vocab = Vocabulary(need_default=False)
    ner_vocab = Vocabulary(need_default=False)
    srl_vocab = Vocabulary(need_default=False)
    chunk_vocab = Vocabulary(need_default=False)
    sets = []

    for file in files:
        sents = []
        pos_labels, ner_labels, srl_labels, chunk_labels = [], [], [], []
        trimmed_sent = 0
        with open(file) as f:
            lines = f.readlines()
            sent = []
            pos_label, ner_label, srl_label, chunk_label = [], [], [], []
            for l in lines:
                l = l.strip()
                if l == '':
                    if len(sent) > 0:
                        if len(sent) > max_len:
                            trimmed_sent += 1
                            pos_labels.append(pos_label[:max_len])
                            ner_labels.append(ner_label[:max_len])
                            srl_labels.append(srl_label[:max_len])
                            chunk_labels.append(chunk_label[:max_len])
                            sents.append(sent[:max_len])
                        else:
                            pos_labels.append(pos_label)
                            ner_labels.append(ner_label)
                            srl_labels.append(srl_label)
                            chunk_labels.append(chunk_label)
                            sents.append(sent)
                        sent = []
                        pos_label, ner_label, srl_label, chunk_label = [], [], [], []
                else:
                    l = l.split()
                    word = l[0]

                    if not keep_case:
                        word = word.lower()

                    sent.append(word)
                    pos_label.append(l[2])
                    ner_label.append(l[3])
                    srl_label.append(l[4])
                    chunk_label.append(l[5])

                    vocab.add_word(word)
                    pos_vocab.add_word(l[2])
                    ner_vocab.add_word(l[3])
                    srl_vocab.add_word(l[4])
                    chunk_vocab.add_word(l[5])

        sets.append({
            'sents': sents,
            'pos_labels': pos_labels,
            'ner_labels': ner_labels,
            'srl_labels': srl_labels,
            'chunk_labels': chunk_labels
        })

        logger.info('Get {} instances from file {}'.format(len(sents), file))
        if trimmed_sent:
            logger.warning('{} sentences are trimmed. Max sentence length: {}.'
                           .format(trimmed_sent, max_len))

    logger.info('Building vocabulary...')
    vocab.build_vocab()
    logger.info('Finished. Size of vocab: {}'.format(len(vocab)))

    pos_vocab.build_vocab()
    ner_vocab.build_vocab()
    srl_vocab.build_vocab()
    chunk_vocab.build_vocab()
    logger.info('# class in POS Tagging: {}'.format(len(pos_vocab)))
    logger.info('# class in NER Tagging: {}'.format(len(ner_vocab)))
    logger.info('# class in SRL Tagging: {}'.format(len(srl_vocab)))
    logger.info('# class in Chunking: {}'.format(len(chunk_vocab)))

    return sets, vocab, [pos_vocab, ner_vocab, srl_vocab, chunk_vocab]


def read_instances_from_test_file(test_file, max_len, keep_case):
    '''
    Collect instances from test file.
    Difference from above: do not add word to vocab.
    '''

    sents = []
    pos_labels, ner_labels, srl_labels, chunk_labels = [], [], [], []
    trimmed_sent = 0
    with open(test_file) as f:
        lines = f.readlines()
        sent = []
        pos_label, ner_label, srl_label, chunk_label = [], [], [], []
        for l in lines:
            l = l.strip()
            if l == '':
                if len(sent) > 0:
                    if len(sent) > max_len:
                        trimmed_sent += 1
                        pos_labels.append(pos_label[:max_len])
                        ner_labels.append(ner_label[:max_len])
                        srl_labels.append(srl_label[:max_len])
                        chunk_labels.append(chunk_label[:max_len])
                        sents.append(sent[:max_len])
                    else:
                        pos_labels.append(pos_label)
                        ner_labels.append(ner_label)
                        srl_labels.append(srl_label)
                        chunk_labels.append(chunk_label)
                        sents.append(sent)
                    sent = []
                    pos_label, ner_label, srl_label, chunk_label = [], [], [], []
            else:
                l = l.split()
                word = l[0]

                if not keep_case:
                    word = word.lower()

                sent.append(word)
                pos_label.append(l[2])
                ner_label.append(l[3])
                srl_label.append(l[4])
                chunk_label.append(l[5])

    test_set = {
        'sents': sents,
        'pos_labels': pos_labels,
        'ner_labels': ner_labels,
        'srl_labels': srl_labels,
        'chunk_labels': chunk_labels
    }

    if trimmed_sent:
        logger.warning('{} sentences are trimmed. Max sentence length: {}'
                       .format(trimmed_sent, max_len))

    logger.info('Get {} instances from file {}'.format(len(sents), test_file))

    return test_set


def convert_to_idx(sets, vocab, lb_vocabs):
    ''' convert token into index using vocab '''

    pos_vocab = lb_vocabs[0]
    ner_vocab = lb_vocabs[1]
    srl_vocab = lb_vocabs[2]
    chunk_vocab = lb_vocabs[3]

    idx_sets = []

    for set in sets:
        sents = set['sents']
        pos_labels = set['pos_labels']
        ner_labels = set['ner_labels']
        srl_labels = set['srl_labels']
        chunk_labels = set['chunk_labels']

        idx_sents = []
        idx_pos_labels = []
        idx_ner_labels = []
        idx_srl_labels = []
        idx_chunk_labels = []

        for sent in sents:
            idx_sent = []
            for word in sent:
                idx_sent.append(vocab.to_index(word))
            idx_sents.append(idx_sent)

        for pos_label in pos_labels:
            idx_label = []
            for lb in pos_label:
                idx_label.append(pos_vocab.to_index(lb))
            idx_pos_labels.append(idx_label)

        for ner_label in ner_labels:
            idx_label = []
            for lb in ner_label:
                idx_label.append(ner_vocab.to_index(lb))
            idx_ner_labels.append(idx_label)

        for srl_label in srl_labels:
            idx_label = []
            for lb in srl_label:
                idx_label.append(srl_vocab.to_index(lb))
            idx_srl_labels.append(idx_label)

        for chunk_label in chunk_labels:
            idx_label = []
            for lb in chunk_label:
                idx_label.append(chunk_vocab.to_index(lb))
            idx_chunk_labels.append(idx_label)

        idx_sets.append({
            'sents': idx_sents,
            'pos_labels': idx_pos_labels,
            'ner_labels': idx_ner_labels,
            'srl_labels': idx_srl_labels,
            'chunk_labels': idx_chunk_labels
        })

    return idx_sets


def main():
    ''' main function '''

    start_time = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-train_file', type=str,
                        default='/remote-home/txsun/data/multi_ontonotes/train.txt')
    parser.add_argument('-dev_file', type=str,
                        default='/remote-home/txsun/data/multi_ontonotes/valid.txt')
    parser.add_argument('-test_file', type=str,
                        default='/remote-home/txsun/data/multi_ontonotes/test.txt')
    parser.add_argument('-save_path', type=str,
                        default='data')
    parser.add_argument('-max_len', type=int, default=400)
    parser.add_argument('-keep_case', type=bool, default=False)

    args = parser.parse_args()

    files = [args.train_file, args.dev_file]
    text_sets, vocab, lb_vocabs = read_instances_from_file(files, args.max_len, args.keep_case)
    text_test_set = read_instances_from_test_file(args.test_file, args.max_len, args.keep_case)
    text_sets.append(text_test_set)

    idx_sets = convert_to_idx(text_sets, vocab, lb_vocabs)

    train_set, dev_set, test_set = idx_sets

    data = {
        'settings': args,
        'vocab': vocab,
        'class_dict': lb_vocabs,
        'train': train_set,
        'dev': dev_set,
        'test': test_set
    }

    logger.info('Testing pre-processing...')
    logger.info('The first example in train set:')
    logger.info(' '.join([vocab.to_word(idx) for idx in train_set['sents'][0]]))
    logger.info(' '.join([lb_vocabs[0].to_word(idx) for idx in train_set['pos_labels'][0]]))
    logger.info(' '.join([lb_vocabs[1].to_word(idx) for idx in train_set['ner_labels'][0]]))
    logger.info(' '.join([lb_vocabs[2].to_word(idx) for idx in train_set['srl_labels'][0]]))
    logger.info(' '.join([lb_vocabs[3].to_word(idx) for idx in train_set['chunk_labels'][0]]))

    logger.info('The first example in dev set:')
    logger.info(' '.join([vocab.to_word(idx) for idx in dev_set['sents'][0]]))
    logger.info(' '.join([lb_vocabs[0].to_word(idx) for idx in dev_set['pos_labels'][0]]))
    logger.info(' '.join([lb_vocabs[1].to_word(idx) for idx in dev_set['ner_labels'][0]]))
    logger.info(' '.join([lb_vocabs[2].to_word(idx) for idx in dev_set['srl_labels'][0]]))
    logger.info(' '.join([lb_vocabs[3].to_word(idx) for idx in dev_set['chunk_labels'][0]]))

    logger.info('The first example in test set:')
    logger.info(' '.join([vocab.to_word(idx) for idx in test_set['sents'][0]]))
    logger.info(' '.join([lb_vocabs[0].to_word(idx) for idx in test_set['pos_labels'][0]]))
    logger.info(' '.join([lb_vocabs[1].to_word(idx) for idx in test_set['ner_labels'][0]]))
    logger.info(' '.join([lb_vocabs[2].to_word(idx) for idx in test_set['srl_labels'][0]]))
    logger.info(' '.join([lb_vocabs[3].to_word(idx) for idx in test_set['chunk_labels'][0]]))

    logger.info('Dumping the processed data to pickle file {}'
                .format(os.path.join(args.save_path, 'all_data.pkl')))
    torch.save(data, os.path.join(args.save_path, 'all_data.pkl'))

    logger.info('Finished. Dumping vocabulary to file {}'
                .format(os.path.join(args.save_path, 'vocab.txt')))
    with open(os.path.join(args.save_path, 'vocab.txt'), mode='w', encoding='utf-8') as f:
        for i in range(len(vocab)):
            f.write(vocab.to_word(i) + '\n')

    logger.info('Finished. Dumping POS tagging labels to file {}'
                .format(os.path.join(args.save_path, 'pos_labels.txt')))
    with open(os.path.join(args.save_path, 'pos_labels.txt'), mode='w', encoding='utf-8') as f:
        for i in range(len(lb_vocabs[0])):
            f.write(lb_vocabs[0].to_word(i) + '\n')

    logger.info('Finished. Dumping NER tagging labels to file {}'
                .format(os.path.join(args.save_path, 'ner_labels.txt')))
    with open(os.path.join(args.save_path, 'ner_labels.txt'), mode='w', encoding='utf-8') as f:
        for i in range(len(lb_vocabs[1])):
            f.write(lb_vocabs[1].to_word(i) + '\n')

    logger.info('Finished. Dumping SRL tagging labels to file {}'
                .format(os.path.join(args.save_path, 'srl_labels.txt')))
    with open(os.path.join(args.save_path, 'srl_labels.txt'), mode='w', encoding='utf-8') as f:
        for i in range(len(lb_vocabs[2])):
            f.write(lb_vocabs[2].to_word(i) + '\n')

    logger.info('Finished. Dumping Chunking labels to file {}'
                .format(os.path.join(args.save_path, 'chunk_labels.txt')))
    with open(os.path.join(args.save_path, 'chunk_labels.txt'), mode='w', encoding='utf-8') as f:
        for i in range(len(lb_vocabs[3])):
            f.write(lb_vocabs[3].to_word(i) + '\n')

    logger.info('Finished. Elapse: {}s.'.format(time.time() - start_time))


if __name__ == '__main__':
    main()