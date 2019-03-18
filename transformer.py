import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from crf import ConditionalRandomField


def calc_loss(logit, y, mask):
    ''' Calculate loss of single task. '''

    criterion = nn.CrossEntropyLoss(reduce='none')

    bsz = logit.size(0)
    seq_len = logit.size(1)

    loss_vec = criterion(logit.reshape(bsz*seq_len, logit.size(2)),
                         y.reshape(bsz*seq_len))
    loss = loss_vec.masked_select(mask.reshape(-1)).mean()

    return loss


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    h = query.size(1)
    d_k = query.size(-1)
    seq_len = query.size(-2)
    mask = mask.expand(-1, -1, seq_len, -1).expand(-1, h, -1, -1)
    # bsz x h x seq_len x seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -np.inf)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            # mask size: batch_size x 1 x 1 x seq_len
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # dim of q,k,v: batch_size x h x seq_len x d_k
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        # dim of x: batch_size x h x seq_len x d_k
        # dim of self.attn: batch_size x h x seq_len x seq_len
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Embeddings(nn.Module):
    def __init__(self, we, vocab, d_model):
        super(Embeddings, self).__init__()
        word_embedding = torch.from_numpy(we).float()
        self.embed = nn.Embedding(vocab, d_model, _weight=word_embedding)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)).float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # dim of pe: 1 x max_len x d_model
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # dim of x: batch_size x seq_len x d_model
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    """
    A transformer encoder.
    """
    def __init__(self, args, config, word_embedding):
        super(Transformer, self).__init__()

        n_classes = args.n_classes
        d_model = int(config['d_model'])
        h = int(config['n_head'])
        d_ff = int(config['d_ff'])
        N = int(config['n_layer'])
        dropout = float(config['dropout'])
        vocab_size = args.vocab_size
        self.use_crf = int(config['use_crf'])

        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        pe = PositionalEncoding(d_model, dropout)
        layer = EncoderLayer(d_model, c(attn), c(ffn), dropout)
        
        self.embed = nn.Sequential(Embeddings(word_embedding, vocab_size,
                                              d_model), c(pe))
        self.encoder = Encoder(c(layer), N)
        self.out = nn.ModuleList([
            nn.Linear(d_model, n_classes[0]),
            nn.Linear(d_model, n_classes[1]),
            nn.Linear(d_model, n_classes[2])
        ])
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if self.use_crf:
            self.crf = nn.ModuleList([
                ConditionalRandomField(n_classes[0], True),
                ConditionalRandomField(n_classes[1], True),
                ConditionalRandomField(n_classes[2], True)
            ])

        self.log_sigma_square_1 = nn.Parameter(torch.Tensor([0]))
        self.log_sigma_square_2 = nn.Parameter(torch.Tensor([0]))
        self.log_sigma_square_3 = nn.Parameter(torch.Tensor([0]))
    
    def forward(self, src, y1, y2, y3, src_mask):
        x = self.encoder(self.embed(src), src_mask.unsqueeze(-2))
        logits = [self.out[i](x) for i in range(3)]
        y = [y1, y2, y3]

        if self.use_crf:
            losses = [self.crf[i](logits[i], y[i], src_mask).mean()
                      for i in range(3)]
            preds = [self.crf[i].viterbi_decode(logits[i], src_mask)
                     for i in range(3)]
        else:
            losses = [calc_loss(logits[i], y[i], src_mask) for i in range(3)]
            preds = [torch.argmax(logits[i], dim=2) for i in range(3)]

        return losses, preds


        