import torch
import torch.utils.data
import numpy as np


def custom_collate(batch):
    ''' pad sentence and labels '''

    DEFAULT_PADDING_LABEL = 0

    sents, pos_labels, ner_labels, chunk_labels = zip(*batch)
    max_len = max(len(sent) for sent in sents)

    batch_sent = [sent + [DEFAULT_PADDING_LABEL]\
                  * (max_len - len(sent)) for sent in sents]

    batch_pos_label = [label + [DEFAULT_PADDING_LABEL] \
                  * (max_len - len(label)) for label in pos_labels]

    batch_ner_label = [label + [DEFAULT_PADDING_LABEL] \
                   * (max_len - len(label)) for label in ner_labels]

    batch_chunk_label = [label + [DEFAULT_PADDING_LABEL] \
                   * (max_len - len(label)) for label in chunk_labels]

    batch_sent = torch.LongTensor(np.array(batch_sent)) # bsz x seq_len
    batch_pos_label = torch.LongTensor(np.array(batch_pos_label))
    batch_ner_label = torch.LongTensor(np.array(batch_ner_label))
    batch_chunk_label = torch.LongTensor(np.array(batch_chunk_label))
    batch_mask = (batch_sent != DEFAULT_PADDING_LABEL).unsqueeze(-2) # bsz x 1 x seq_len

    return batch_sent, batch_pos_label, batch_ner_label, batch_chunk_label, batch_mask


class SeqLabDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.sents = dataset['sents']
        self.pos_labels = dataset['pos_labels']
        self.ner_labels = dataset['ner_labels']
        self.chunk_labels = dataset['chunk_labels']

    def __getitem__(self, item):
        return self.sents[item], self.pos_labels[item],\
               self.ner_labels[item], self.chunk_labels[item]

    def __len__(self):
        return len(self.sents)