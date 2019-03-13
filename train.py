import argparse
import os
import sys
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from utils import load_word_emb
from dataset import SeqLabDataset, custom_collate
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from seqeval.metrics import f1_score
from seqeval.metrics import classification_report

summary_writer = None
lb_vocabs = None
steps = 0

logger = logging.getLogger('train')
logger.setLevel(level=logging.DEBUG)

# Stream Handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

# File Handler
file_handler = logging.FileHandler('logs/train.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(fmt='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
                                   datefmt='%Y/%m/%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


def calc_stl_loss(logit, y, mask):
    ''' Calculate loss of single task. '''

    criterion = nn.CrossEntropyLoss(reduce='none')

    bsz = logit.size(0)
    seq_len = logit.size(1)

    loss_vec = criterion(logit.reshape(bsz*seq_len, logit.size(2)),
                         y.reshape(bsz*seq_len))
    loss = loss_vec.masked_select(mask.reshape(-1)).mean()

    return loss


def calc_avg_loss(logits, ys, mask):
    ''' Average loss of three tasks (POS, NER, Chunking) '''

    criterion = nn.CrossEntropyLoss(reduce='none')

    logit1, logit2, logit3 = logits
    y1, y2, y3 = ys
    bsz = logit1.size(0)
    seq_len = logit1.size(1)

    loss1_vec = criterion(logit1.reshape(bsz * seq_len, logit1.size(2)),
                          y1.reshape(bsz * seq_len))
    loss2_vec = criterion(logit2.reshape(bsz * seq_len, logit2.size(2)),
                          y2.reshape(bsz * seq_len))
    loss3_vec = criterion(logit3.reshape(bsz * seq_len, logit3.size(2)),
                          y3.reshape(bsz * seq_len))

    loss1 = loss1_vec.masked_select(mask.reshape(-1)).mean()
    loss2 = loss2_vec.masked_select(mask.reshape(-1)).mean()
    loss3 = loss3_vec.masked_select(mask.reshape(-1)).mean()

    loss = (loss1 + loss2 + loss3) / 3

    return loss


def train_epoch(model, train_iter, optimizer, accumulation_steps):

    global steps
    global lb_vocabs
    global summary_writer

    total_loss = 0

    # eval POS tagging: Accuracy
    all_correct, corrects = 0, 0
    all_instance, instances = 0, 0

    # eval NER: F1
    all_ner_pred_lst, ner_pred_lst = [], []
    all_ner_true_lst, ner_true_lst = [], []

    # eval Chunking: F1
    all_chunk_pred_lst, chunk_pred_lst = [], []
    all_chunk_true_lst, chunk_true_lst = [], []

    pos_vocab, ner_vocab, chunk_vocab = lb_vocabs

    print_every = 128

    model.train()

    for batch in train_iter:

        steps += 1

        x, y1, y2, y3, mask = batch
        x, y1, y2, y3, mask = x.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), mask.cuda()

        logit1, logit2, logit3 = model(x, mask)
        # shape of logit: batch_size x seq_len x n_class

        # loss = calc_avg_loss([logit1, logit2, logit3], [y1, y2, y3], mask) / accumulation_steps
        loss = calc_stl_loss(logit1, y1, mask)

        loss.backward()

        total_loss += loss.item()

        pos_pred = torch.argmax(logit1, dim=2)
        ner_pred = torch.argmax(logit2, dim=2)
        chunk_pred = torch.argmax(logit3, dim=2)
        # bsz x seq_len

        for i in range(pos_pred.size(0)):
            i_pos_pred = pos_pred[i].masked_select(mask[i].reshape(-1))
            i_pos_true = y1[i].masked_select(mask[i].reshape(-1))
            all_correct += (i_pos_pred.data == i_pos_true.data).sum()
            corrects += (i_pos_pred.data == i_pos_true.data).sum()
            all_instance += i_pos_pred.size(0)
            instances += i_pos_pred.size(0)

            i_ner_pred = ner_pred[i].masked_select(mask[i].reshape(-1))
            i_ner_true = y2[i].masked_select(mask[i].reshape(-1))
            ner_pred_lst += [ner_vocab.to_word(int(val)) for val in i_ner_pred]
            all_ner_pred_lst += [ner_vocab.to_word(int(val)) for val in i_ner_pred]
            ner_true_lst += [ner_vocab.to_word(int(val)) for val in i_ner_true]
            all_ner_true_lst += [ner_vocab.to_word(int(val)) for val in i_ner_true]

            i_chunk_pred = chunk_pred[i].masked_select(mask[i].reshape(-1))
            i_chunk_true = y3[i].masked_select(mask[i].reshape(-1))
            chunk_pred_lst += [chunk_vocab.to_word(int(val)) for val in i_chunk_pred]
            all_chunk_pred_lst += [chunk_vocab.to_word(int(val)) for val in i_chunk_pred]
            chunk_true_lst += [chunk_vocab.to_word(int(val)) for val in i_chunk_true]
            all_chunk_true_lst += [chunk_vocab.to_word(int(val)) for val in i_chunk_true]

        if steps % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if steps % print_every == 0:
                pos_acc = float(corrects) / float(instances)
                ner_f1 = f1_score(ner_true_lst, ner_pred_lst)
                chunk_f1 = f1_score(chunk_true_lst, chunk_pred_lst)

                summary_writer.add_scalar('loss', total_loss, steps)
                summary_writer.add_scalars('train_measure', {
                    'POS-ACC': pos_acc,
                    'NER-F1': ner_f1,
                    'CHUNK-F1': chunk_f1
                }, steps)
                logger.info('   - Step {}: loss {}, pos_acc {:.3f}%, ner_f1 {:.3f}%, chunk_f1 {:.3f}%'
                            .format(steps, total_loss, pos_acc*100, ner_f1*100, chunk_f1*100))

                total_loss = 0
                instances, corrects = 0, 0
                ner_pred_lst, ner_true_lst = [], []
                chunk_pred_lst, chunk_true_lst = [], []

    all_pos_acc = float(all_correct) / float(all_instance)
    all_ner_f1 = f1_score(all_ner_true_lst, all_ner_pred_lst)
    all_chunk_f1 = f1_score(all_chunk_true_lst, all_chunk_pred_lst)

    return all_pos_acc, all_ner_f1, all_chunk_f1


def eval_epoch(model, data_iter):

    global lb_vocabs
    pos_vocab, ner_vocab, chunk_vocab = lb_vocabs

    logger.info('Evaluating...')

    total_loss = 0

    # eval POS tagging: Accuracy
    all_correct = 0
    all_instance = 0

    # eval NER: F1
    all_ner_pred_lst = []
    all_ner_true_lst = []

    # eval Chunking: F1
    all_chunk_pred_lst = []
    all_chunk_true_lst = []

    model.eval()

    with torch.no_grad():
        for batch in data_iter:
            x, y1, y2, y3, mask = batch
            x, y1, y2, y3, mask = x.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), mask.cuda()

            logit1, logit2, logit3 = model(x, mask)
            # shape of logit: batch_size x seq_len x n_class

            loss = calc_avg_loss([logit1, logit2, logit3], [y1, y2, y3], mask)

            total_loss += loss.item()

            pos_pred = torch.argmax(logit1, dim=2)
            ner_pred = torch.argmax(logit2, dim=2)
            chunk_pred = torch.argmax(logit3, dim=2)
            # bsz x seq_len

            for i in range(pos_pred.size(0)):
                # eval POS Tagging: Accuracy
                i_pos_pred = pos_pred[i].masked_select(mask[i].reshape(-1))
                i_pos_true = y1[i].masked_select(mask[i].reshape(-1))
                all_correct += (i_pos_pred.data == i_pos_true.data).sum()
                all_instance += i_pos_pred.size(0)

                # eval NER: F1
                i_ner_pred = ner_pred[i].masked_select(mask[i].reshape(-1))
                i_ner_true = y2[i].masked_select(mask[i].reshape(-1))
                all_ner_pred_lst += [ner_vocab.to_word(int(val)) for val in i_ner_pred]
                all_ner_true_lst += [ner_vocab.to_word(int(val)) for val in i_ner_true]

                # eval Chunking: F1
                i_chunk_pred = chunk_pred[i].masked_select(mask[i].reshape(-1))
                i_chunk_true = y3[i].masked_select(mask[i].reshape(-1))
                all_chunk_pred_lst += [chunk_vocab.to_word(int(val)) for val in i_chunk_pred]
                all_chunk_true_lst += [chunk_vocab.to_word(int(val)) for val in i_chunk_true]

    all_pos_acc = float(all_correct) / float(all_instance)
    all_ner_f1 = f1_score(all_ner_true_lst, all_ner_pred_lst)
    all_chunk_f1 = f1_score(all_chunk_true_lst, all_chunk_pred_lst)

    logger.info('Finished.')
    logger.info('NER Report:\n'+classification_report(all_ner_true_lst, all_ner_pred_lst))
    logger.info('Chunking Report:\n'+classification_report(all_chunk_true_lst, all_chunk_pred_lst))

    return total_loss, all_pos_acc, all_ner_f1, all_chunk_f1


def train(model, train_iter, dev_iter, optimizer, device, args):
    ''' start training '''

    total_time = time.time()
    logger.info('Training......')
    model = nn.DataParallel(model, device).cuda()

    log_path = args.log_dir + str(time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime()))
    global summary_writer
    summary_writer = SummaryWriter(log_path)

    for i_epoch in range(args.n_epoch):
        logger.info('Epoch {}'.format(i_epoch + 1))

        start_time = time.time()
        pos_acc, ner_f1, chunk_f1 = train_epoch(model, train_iter, optimizer, args.accumulate_grad)
        logger.info(' Epoch {} finished. Measure: {:.3f}%, {:.3f}%, {:.3f}%. Elapse: {}s.'
                    .format(i_epoch+1, pos_acc, ner_f1,
                            chunk_f1, time.time() - start_time))

        start_time = time.time()
        dev_loss, dev_pos, dev_ner, dev_chunk = eval_epoch(model, dev_iter)
        logger.info('Validation: Loss: {}  Measure: {:.3f}%, {:.3f}%, {:.3f}%. Elapse: {}s.'
                    .format(dev_loss, dev_pos, dev_ner, dev_chunk,
                            time.time() - start_time))

        summary_writer.add_scalars('dev_measure', {
            'LOSS': dev_loss,
            'POS-ACC': dev_pos,
            'NER-F1': dev_ner,
            'CHUNK-F1': dev_chunk
        }, i_epoch+1)

    summary_writer.export_scalars_to_json('save/result_' +
        str(time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime()))
        + '.json')
    logger.info('Results saved to save/ directory.')
    summary_writer.close()
    del summary_writer

    logger.info('Training finished. Cost {} hours.'.format((time.time() - total_time)/3600))
    logger.info('Dumping model...')
    torch.save(model, 'save/model' +
               str(time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime()))
               + '.pkl')
    logger.info('Model saved.')

    return


def test(model, test_iter):
    logger.info('Testing...')
    model = model.cuda()
    test_acc = eval_epoch(model, test_iter)
    return test_acc


def main():
    ''' main function '''

    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 2, 3'
    gpu = [0, 1]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n_epoch', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-accumulate_grad', type=int, default=1)
    parser.add_argument('-d_model', type=int, default=300)
    parser.add_argument('-d_ff', type=int, default=512)
    parser.add_argument('-n_head', type=int, default=6)
    parser.add_argument('-n_layer', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-early_stop', type=int, default=3)
    parser.add_argument('-log_dir', type=str, default='logs/tensorboardlogs/')
    parser.add_argument('-embed_path', type=str,
                        default='/remote-home/txsun/data/word-embedding/glove/glove.6B.300d.txt')

    args = parser.parse_args()

    bsz = args.batch_size // args.accumulate_grad
    logger.info('========== Loading Datasets ==========')
    data = torch.load('data/all_data.pkl')
    vocab = data['vocab']
    args.vocab_size = len(vocab)

    global lb_vocabs
    lb_vocabs = data['class_dict']
    del lb_vocabs[2]
    args.n_classes = [len(lb_voc) for lb_voc in lb_vocabs]

    logger.info('# POS Tagging labels: {}'.format(args.n_classes[0]))
    logger.info('# NER Tagging labels: {}'.format(args.n_classes[1]))
    logger.info('# Chunking labels: {}'.format(args.n_classes[2]))
    assert len(args.n_classes) == 3
    train_data = data['train']
    dev_data = data['dev']
    test_data = data['test']

    train_set = SeqLabDataset(train_data)
    train_iter = DataLoader(train_set, batch_size=bsz, drop_last=True,
                            shuffle=True, num_workers=2, collate_fn=custom_collate)
    logger.info('Train set loaded.')

    dev_set = SeqLabDataset(dev_data)
    dev_iter = DataLoader(dev_set, batch_size=bsz,
                          num_workers=2, collate_fn=custom_collate)
    logger.info('Development set loaded.')

    test_set = SeqLabDataset(test_data)
    test_iter = DataLoader(test_set, batch_size=bsz,
                           num_workers=2, collate_fn=custom_collate)
    logger.info('Test set loaded.')
    logger.info('Datasets finished.')

    logger.info('====== Loading Word Embedding =======')
    word_embedding = load_word_emb(args.embed_path, args.d_model, vocab)

    logger.info('========== Preparing Model ==========')
    transformer = Transformer(args, word_embedding)
    params = list(transformer.named_parameters())
    for name, param in params:
        logger.info('{}: {}'.format(name, param.shape))
    logger.info('# Parameters: {}.'.format(sum(param.numel() for param in transformer.parameters())))

    logger.info('========== Training Model ==========')
    opt = optim.Adam(transformer.parameters(), lr=5e-5)
    train(transformer, train_iter, dev_iter, opt, gpu, args)


if __name__ == '__main__':
    main()