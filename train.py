import argparse
import os
import sys
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from lstm import BiLSTM
from utils import load_word_emb
from dataset import SeqLabDataset, custom_collate
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from seqeval.metrics import f1_score
from seqeval.metrics import classification_report

summary_writer = None
lb_vocabs = None
steps = 0
model_config = {}

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


# def calc_stl_loss(logit, y, mask):
#     ''' Calculate loss of single task. '''
#
#     criterion = nn.CrossEntropyLoss(reduce='none')
#
#     bsz = logit.size(0)
#     seq_len = logit.size(1)
#
#     loss_vec = criterion(logit.reshape(bsz*seq_len, logit.size(2)),
#                          y.reshape(bsz*seq_len))
#     loss = loss_vec.masked_select(mask.reshape(-1)).mean()
#
#     return loss
#
#
# def calc_avg_loss(logits, ys, mask):
#     ''' Average loss of three tasks (POS, NER, Chunking) '''
#
#     criterion = nn.CrossEntropyLoss(reduce='none')
#
#     logit1, logit2, logit3 = logits
#     y1, y2, y3 = ys
#     bsz = logit1.size(0)
#     seq_len = logit1.size(1)
#
#     loss1_vec = criterion(logit1.reshape(bsz * seq_len, logit1.size(2)),
#                           y1.reshape(bsz * seq_len))
#     loss2_vec = criterion(logit2.reshape(bsz * seq_len, logit2.size(2)),
#                           y2.reshape(bsz * seq_len))
#     loss3_vec = criterion(logit3.reshape(bsz * seq_len, logit3.size(2)),
#                           y3.reshape(bsz * seq_len))
#
#     loss1 = loss1_vec.masked_select(mask.reshape(-1)).mean()
#     loss2 = loss2_vec.masked_select(mask.reshape(-1)).mean()
#     loss3 = loss3_vec.masked_select(mask.reshape(-1)).mean()
#
#     loss = (loss1 + loss2 + loss3) / 3
#
#     return loss


def train_epoch(model, train_iter, optimizer, args):

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

    loss_split = args.loss_split.split('-')
    loss_split = [int(loss_split[i]) for i in range(3)]
    sum = 0
    for i in range(3):
        sum += loss_split[i]
    loss_split = [loss_split[i] / sum for i in range(3)]

    model.train()

    for batch in train_iter:

        steps += 1

        x, y1, y2, y3, mask = batch
        x, y1, y2, y3, mask = x.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), mask.cuda()

        losses, preds = model(x, y1, y2, y3, mask)

        pos_loss, ner_loss, chunk_loss = losses
        pos_pred, ner_pred, chunk_pred = preds

        loss = (loss_split[0] * pos_loss + loss_split[1] * ner_loss + loss_split[2] * chunk_loss)\
               / args.accumulation_steps

        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)

        total_loss += loss.item()

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

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if steps % print_every == 0:
                pos_acc = float(corrects) / float(instances)
                ner_f1 = f1_score(ner_true_lst, ner_pred_lst)
                chunk_f1 = f1_score(chunk_true_lst, chunk_pred_lst)

                summary_writer.add_scalar('Train/train_loss', total_loss, steps)
                summary_writer.add_scalars('Train/train_measure', {
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

            losses, preds = model(x, y1, y2, y3, mask)
            pos_loss, ner_loss, chunk_loss = losses
            pos_pred, ner_pred, chunk_pred = preds

            loss = (pos_loss + ner_loss + chunk_loss) / 3
            # loss = pos_loss / accumulation_steps
            total_loss += loss.item()

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


def train(model, train_iter, dev_iter, optimizer, args):
    ''' start training '''

    total_time = time.time()
    logger.info('Training......')
    # model = nn.DataParallel(model, device).cuda()
    model = model.cuda()

    config_str = ''
    global model_config
    for key, value in model_config.items():
        config_str += key + '-' + value + '-'
    log_path = os.path.join(args.log_dir, config_str + args.loss_split)
    # log_path = args.log_dir + str(time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime()))
    global summary_writer
    summary_writer = SummaryWriter(log_path)

    for i_epoch in range(args.n_epoch):
        logger.info('Epoch {}'.format(i_epoch + 1))

        start_time = time.time()
        pos_acc, ner_f1, chunk_f1 = train_epoch(model, train_iter, optimizer, args)
        logger.info(' Epoch {} finished. Measure: {:.3f}%, {:.3f}%, {:.3f}%. Elapse: {:.3f}%s.'
                    .format(i_epoch+1, pos_acc*100, ner_f1*100,
                            chunk_f1*100, time.time() - start_time))

        start_time = time.time()
        dev_loss, dev_pos, dev_ner, dev_chunk = eval_epoch(model, dev_iter)
        logger.info('Validation: Loss: {}  Measure: {:.3f}%, {:.3f}%, {:.3f}%. Elapse: {:.3f}%s.'
                    .format(dev_loss, dev_pos*100, dev_ner*100, dev_chunk*100,
                            time.time() - start_time))

        summary_writer.add_scalar('Validation/dev_loss', dev_loss, i_epoch + 1)
        summary_writer.add_scalars('Validation/dev_measure', {
            'POS-ACC': dev_pos,
            'NER-F1': dev_ner,
            'CHUNK-F1': dev_chunk
        }, i_epoch+1)

    # summary_writer.export_scalars_to_json('save/result_' +
    #     str(time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime()))
    #     + '.json')
    # logger.info('Results saved to save/ directory.')
    summary_writer.close()
    del summary_writer

    logger.info('Training finished. Cost {:.3f}% hours.'.format((time.time() - total_time)/3600))
    logger.info('Dumping model...')
    save_path = os.path.join(args.save_path, config_str + args.loss_split)
    torch.save(model, save_path + '.pkl')
    logger.info('Model saved as {}.'.format(save_path + '.pkl'))

    return


def test(model, test_iter):
    logger.info('Testing...')
    model = model.cuda()
    test_acc = eval_epoch(model, test_iter)
    return test_acc


def main():
    ''' main function '''

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n_epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-accumulation_steps', type=int, default=1)
    parser.add_argument('-model_config', type=str, default='lstm.config')
    parser.add_argument('-loss_split', type=str, default='1-1-1')
    parser.add_argument('-log_dir', type=str, default='logs/tensorboardlogs/')
    parser.add_argument('-save_path', type=str, default='saved_models/')
    parser.add_argument('-embed_path', type=str,
                        default='/remote-home/txsun/data/word-embedding/glove/glove.6B.300d.txt')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    bsz = args.batch_size // args.accumulation_steps
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
    word_embedding = load_word_emb(args.embed_path, 300, vocab)

    logger.info('========== Preparing Model ==========')
    global model_config
    model_config = {}
    logger.info('Reading configure file {}...'.format(args.model_config))
    with open(args.model_config, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key = line.split(':')[0].strip()
            value = line.split(':')[1].strip()
            model_config[key] = value
            logger.info('{}: {}'.format(key, value))

    if model_config['model'] == 'transformer':
        model = Transformer(args, model_config, word_embedding)
    elif model_config['model'] == 'LSTM':
        model = BiLSTM(args, model_config, word_embedding)
    else:
        logger.error('No support for {}.'.format(model_config['model']))
        return

    logger.info('Model parameters:')
    params = list(model.named_parameters())
    for name, param in params:
        logger.info('{}: {}'.format(name, param.shape))
    logger.info('# Parameters: {}.'.format(sum(param.numel() for param in model.parameters())))

    logger.info('========== Training Model ==========')
    opt = optim.Adam(model.parameters(), lr=float(model_config['lr']))
    train(model, train_iter, dev_iter, opt, args)

    return


if __name__ == '__main__':
    main()