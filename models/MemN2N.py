import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim

import torch.nn.functional as F
from utils.masked_cross_entropy import *
from utils.config import *
import random
import numpy as np
import datetime

from utils.measures import wer, moses_multi_bleu
import os
from sklearn.metrics import f1_score

import nsml


class MemN2N(nn.Module):
    def __init__(self, hidden_size, max_len, max_r, lang, path, task, lr, n_layers, dropout):
        super(MemN2N, self).__init__()
        self.name = "MemN2N"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len  ## max input
        self.max_r = max_r  ## max responce len
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th')
                self.decoder = torch.load(str(path) + '/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                self.decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)
        else:
            self.encoder = EncoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout)
            self.decoder = DecoderRNN(hidden_size, lang.n_words, n_layers, self.dropout)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_gate = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_vac = self.loss_vac / self.print_every
        self.print_every += 1
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg, print_loss_vac, print_loss_ptr)

    def save_model(self, dec_type):
        name_data = "KVR/" if self.task == '' else "BABI/"
        directory = 'save/mem2seq-' + name_data + str(self.task) + 'HDD' + str(self.hidden_size) + 'BSZ' + str(
            args['batch']) + 'DR' + str(self.dropout) + 'L' + str(self.n_layers) + 'lr' + str(self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

    def train_batch(self, input_batches, input_lengths, target_batches,
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio,reset):
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.loss_gate = 0
        self.print_every = 1


        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab, loss_Ptr = 0, 0



        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = (encoder_hidden[0].unsqueeze(0), encoder_hidden[1].unsqueeze(0))

        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(max_target_length, batch_size, encoder_outputs.size(1)))
        all_decoder_outputs_gate = Variable(torch.zeros(max_target_length, batch_size))
        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        if use_teacher_forcing:
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, gate, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr.transpose(0, 1)
                all_decoder_outputs_gate[t] = gate.transpose(0, 1)
                decoder_input = target_batches[t]  # Next input is current target
                if USE_CUDA: decoder_input = decoder_input.cuda()

        else:
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, gate, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr.transpose(0, 1)  #
                all_decoder_outputs_gate[t] = gate.transpose(0, 1)  #
                topv, topvi = decoder_vacab.data.topk(1)
                topp, toppi = decoder_ptr.data.topk(1)
                ## get the correspective word in input
                top_ptr_i = torch.gather(input_batches[:,:,0], 0, toppi.view(1,-1))
                next_in = [top_ptr_i.squeeze()[i].data.item() if (gate.squeeze()[i].data.item() >= 0.5) else topvi.squeeze()[i]
                           for i in range(batch_size)]
                decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()

        # Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),  # -> batch x seq
            target_index.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss_gate = self.criterion(all_decoder_outputs_gate, target_gate.float())

        loss = loss_Vocab + loss_Ptr + loss_gate
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.data.item()
        self.loss_gate += loss_gate.data.item()
        self.loss_ptr += loss_Ptr.data.item()
        self.loss_vac += loss_Vocab.data.item()

    def evaluate_batch(self,batch_size,input_batches, input_lengths, target_batches, target_lengths, target_index,target_gate,src_plain):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches)
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = (encoder_hidden[0].unsqueeze(0),encoder_hidden[1].unsqueeze(0))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.decoder.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(self.max_r, batch_size, encoder_outputs.size(1)))
        all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()
        p = []
        for elm in src_plain:
            elm_temp = [ word_triple[0] for word_triple in elm ]
            p.append(elm_temp)
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_ptr,decoder_vacab,gate,decoder_hidden  = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs_vocab[t] = decoder_vacab
            all_decoder_outputs_ptr[t] = decoder_ptr.transpose(0,1)
            all_decoder_outputs_gate[t] = gate.transpose(0,1)

            topv, topvi = decoder_vacab.data.topk(1)
            topp, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches[:,:,0],0,toppi.view(1, -1))
            next_in = [top_ptr_i.squeeze()[i].data.item() if(gate.squeeze()[i].data.item()>=0.5) else topvi.squeeze()[i] for i in range(batch_size)]
            decoder_input = Variable(torch.LongTensor(next_in))
            # Next input is chosen word
            if USE_CUDA: decoder_input = decoder_input.cuda()

            temp = []
            for i in range(batch_size):
                if(gate.squeeze()[i].data.item()>=0.5):
                    if(toppi.squeeze()[i] >= len(p[i]) ):
                        temp.append('<EOS>')
                    else:
                        temp.append(p[i][toppi.squeeze()[i]])
                else:
                    ind = topvi.squeeze()[i]
                    if ind == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(self.lang.index2word[ind.item()])
            decoded_words.append(temp)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        return decoded_words


    def evaluate(self, dev, avg_best, epoch, BLEU=False):
        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        microF1_PRED, microF1_PRED_cal, microF1_PRED_nav, microF1_PRED_wet = [], [], [], []
        microF1_TRUE, microF1_TRUE_cal, microF1_TRUE_nav, microF1_TRUE_wet = [], [], [], []
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        dialog_acc_dict = {}

        if nsml.IS_ON_NSML:
            pbar = tqdm(enumerate(dev), total=len(dev))
        else:
            pbar = enumerate(dev)

        for j, data_dev in pbar:
            if args['dataset'] == 'kvr':
                words = self.evaluate_batch(len(data_dev[1]), data_dev[0], data_dev[1],
                                            data_dev[2], data_dev[3], data_dev[4], data_dev[5], data_dev[6])
            else:
                words = self.evaluate_batch(len(data_dev[1]), data_dev[0], data_dev[1],
                                            data_dev[2], data_dev[3], data_dev[4], data_dev[5], data_dev[6])
            # acc_P += acc_ptr
            # acc_V += acc_vac
            acc = 0
            w = 0
            temp_gen = []

            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e == '<EOS>':
                        break
                    else:
                        st += e + ' '
                temp_gen.append(st)
                try:
                    correct = data_dev[7][i]
                except:
                    correct = ' '
                ### compute F1 SCORE
                if args['dataset'] == 'kvr':
                    f1_true, f1_pred = computeF1(data_dev[8][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE += f1_true
                    microF1_PRED += f1_pred
                    f1_true, f1_pred = computeF1(data_dev[9][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE_cal += f1_true
                    microF1_PRED_cal += f1_pred
                    f1_true, f1_pred = computeF1(data_dev[10][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE_nav += f1_true
                    microF1_PRED_nav += f1_pred
                    f1_true, f1_pred = computeF1(data_dev[11][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE_wet += f1_true
                    microF1_PRED_wet += f1_pred
                elif args['dataset'] == 'babi' and int(self.task) == 6:
                    f1_true, f1_pred = computeF1(data_dev[-2][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE += f1_true
                    microF1_PRED += f1_pred

                if args['dataset'] == 'babi':
                    if data_dev[-1][i] not in dialog_acc_dict.keys():
                        dialog_acc_dict[data_dev[-1][i]] = []
                    if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                        acc += 1
                        dialog_acc_dict[data_dev[-1][i]].append(1)
                    else:
                        print("Correct:" + str(correct.lstrip().rstrip()))
                        print("XPredict:" + str(st.lstrip().rstrip()))
                        dialog_acc_dict[data_dev[-1][i]].append(0)
                else:
                    if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                        acc += 1
                    else:
                        print("Correct:" + str(correct.lstrip().rstrip()))
                        print("XPredict:" + str(st.lstrip().rstrip()))
                #    print("Correct:"+str(correct.lstrip().rstrip()))
                #    print("\tPredict:"+str(st.lstrip().rstrip()))
                #    print("\tFrom:"+str(self.from_whichs[:,i]))

                w += wer(correct.lstrip().rstrip(), st.lstrip().rstrip())
                ref.append(str(correct.lstrip().rstrip()))
                hyp.append(str(st.lstrip().rstrip()))
                ref_s += str(correct.lstrip().rstrip()) + "\n"
                hyp_s += str(st.lstrip().rstrip()) + "\n"

            acc_avg += acc / float(len(data_dev[1]))
            wer_avg += w / float(len(data_dev[1]))
            # pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg/float(len(dev)),wer_avg/float(len(dev))))

        # dialog accuracy
        if args['dataset'] == 'babi':
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            logging.info("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))

        if args['dataset'] == 'kvr':
            logging.info("F1 SCORE:\t" + str(f1_score(microF1_TRUE, microF1_PRED, average='micro')))
            logging.info("F1 CAL:\t" + str(f1_score(microF1_TRUE_cal, microF1_PRED_cal, average='micro')))
            logging.info("F1 WET:\t" + str(f1_score(microF1_TRUE_wet, microF1_PRED_wet, average='micro')))
            logging.info("F1 NAV:\t" + str(f1_score(microF1_TRUE_nav, microF1_PRED_nav, average='micro')))
        elif args['dataset'] == 'babi' and int(self.task) == 6:
            logging.info("F1 SCORE:\t" + str(f1_score(microF1_TRUE, microF1_PRED, average='micro')))

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        logging.info("BLEU SCORE:" + str(bleu_score))
        if (BLEU):
            if (bleu_score >= avg_best):
                self.save_model(str(self.name) + str(bleu_score))
                logging.info("MODEL SAVED")
            return bleu_score
        else:
            acc_avg = acc_avg / float(len(dev))
            if (acc_avg >= avg_best):
                if nsml.IS_ON_NSML:
                    nsml.save(epoch)
                else:
                    self.save_model(str(self.name) + str(acc_avg))
                    logging.info("MODEL SAVED")
            return acc_avg


def computeF1(entity, st, correct):
    y_pred = [0 for z in range(len(entity))]
    y_true = [1 for z in range(len(entity))]
    for k in st.lstrip().rstrip().split(' '):
        if (k in entity):
            y_pred[entity.index(k)] = 1
    return y_true, y_pred


class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        for hop in range(self.max_hops + 1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        self.softmax = nn.Softmax(dim=1)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(bsz, self.embedding_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.embedding_dim))

    def forward(self, story):
        story = story.transpose(0, 1)
        story_size = story.size()  # b * m * 3
        u = [self.get_state(story.size(0))]

        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1).long())  # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob = self.softmax(torch.sum(m_A * u_temp, 2))
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            m_C = torch.sum(embed_C, 2).squeeze(2)

            prob = prob.unsqueeze(2).expand_as(m_C)
            o = m_C * prob
            o_k = torch.sum(o, 1)

            u_k = u[-1] + o_k
            u.append(u_k) # b * e

        a_hat = u[-1]@self.C[self.max_hops].weight.transpose(0, 1) # b * v
        return o, u

import math
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size ### Vocab size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.cell = nn.LSTM(2*hidden_size, hidden_size, n_layers, dropout=dropout)
        self.W1 = nn.Linear(2*hidden_size, hidden_size)
        v = torch.rand(hidden_size)
        stdv = 1. / math.sqrt(v.size(0))
        v = v.data.normal_(mean=0, std=stdv)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.U = nn.Linear(hidden_size, output_size)
        self.W = nn.Linear(hidden_size, 1)

        if USE_CUDA:
            self.embedding = self.embedding.cuda()
            self.embedding_dropout = self.embedding_dropout.cuda()
            self.cell = self.cell.cuda()
            self.W1 = self.W1.cuda()
            v = v.cuda()
            self.U = self.U.cuda()
            self.W = self.W.cuda()

        self.v = nn.Parameter(v)

    def get_state(self, input):
        c0_encoder = input[0].unsqueeze(0)
        h0_encoder = input[1].unsqueeze(0)
        return (h0_encoder, c0_encoder)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # b , 1 * h, b * v
        # Note: we run this one step at a time
        # Get the embedding of the current input word (last output word)
        max_len = encoder_outputs.size(1)
        batch_size = input_seq.size(0)
        input_seq = input_seq

        word_embedded = self.embedding(input_seq)  # S=1 x B x N
        word_embedded = self.embedding_dropout(word_embedded)

        ## ATTENTION CALCULATION
        s_t = last_hidden[0][-1].unsqueeze(0)
        H = s_t.repeat(max_len, 1, 1).transpose(0,1)
        energy = F.tanh(self.W1(torch.cat([H, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)

        # NORMALIZATION
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        p_ptr = torch.bmm(v, energy)  # [B*1*T]

        a = F.softmax(p_ptr)
        context = a.bmm(encoder_outputs)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context.squeeze()), 1).unsqueeze(0)
        output, hidden = self.cell(rnn_input, last_hidden)

        p_vacab = self.U(output)

        gate = F.sigmoid(self.W(hidden[0][-1]))
        return p_ptr, p_vacab, gate, hidden


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
