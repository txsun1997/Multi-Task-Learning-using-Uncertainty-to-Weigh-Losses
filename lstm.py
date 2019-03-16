import torch
import torch.nn as nn
import numpy as np
from crf import ConditionalRandomField
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def calc_loss(logit, y, mask):
    ''' Calculate loss of single task. '''

    criterion = nn.CrossEntropyLoss(reduce='none')

    bsz = logit.size(0)
    seq_len = logit.size(1)

    loss_vec = criterion(logit.reshape(bsz*seq_len, logit.size(2)),
                         y.reshape(bsz*seq_len))
    loss = loss_vec.masked_select(mask.reshape(-1)).mean()

    return loss


def init_embedding(input_embedding, seed=1337):
    """initiate weights in embedding layer
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)


def init_linear(input_linear, seed=1337):
    """initiate weights in linear layer
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -scope, scope)
    # nn.init.uniform(input_linear.bias, -scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


class BiLSTM(nn.Module):

    def __init__(self, args, config, word_embedding):
        super(BiLSTM, self).__init__()

        self.vocab_size = args.vocab_size
        self.embed_dim = int(config['embed_dim'])
        self.hidden_size = int(config['hidden_size'])
        self.n_classes = args.n_classes
        self.dropout_p = float(config['dropout'])
        self.n_layer = int(config['n_layer'])
        self.use_crf = int(config['use_crf'])

        we = torch.from_numpy(word_embedding).float()
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim, _weight=we)
        self.dropout = nn.Dropout(self.dropout_p)
        self.bilstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size,
                              num_layers=self.n_layer, batch_first=True, bidirectional=True)
        self.out = nn.ModuleList([
            nn.Linear(self.hidden_size * 2, self.n_classes[0]),
            nn.Linear(self.hidden_size * 2, self.n_classes[1]),
            nn.Linear(self.hidden_size * 2, self.n_classes[2])
        ])

        init_linear(self.out[0])
        init_linear(self.out[1])
        init_linear(self.out[2])

        if self.use_crf:
            self.crf = nn.ModuleList([
                ConditionalRandomField(self.n_classes[0], True),
                ConditionalRandomField(self.n_classes[1], True),
                ConditionalRandomField(self.n_classes[2], True)
            ])

    def forward(self, x, y1, y2, y3, mask):
        x = self.embed(x)
        rnn_inp = self.dropout(x)
        rnn_out, _ = self.bilstm(rnn_inp)
        rnn_out = self.dropout(rnn_out)
        logits = [self.out[i](rnn_out) for i in range(3)]
        y = [y1, y2, y3]

        if self.use_crf:
            losses = [self.crf[i](logits[i], y[i], mask).mean()
                      for i in range(3)]
            preds = [self.crf[i].viterbi_decode(logits[i], mask)
                     for i in range(3)]
        else:
            losses = [calc_loss(logits[i], y[i], mask) for i in range(3)]
            preds = [torch.argmax(logits[i], dim=2) for i in range(3)]

        return losses, preds