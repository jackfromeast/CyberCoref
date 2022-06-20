import sys 
sys.path.append("..") 
import torch
import torch.nn as nn
from transformers import BertModel
import attr
from utils import *
from dataLoader import Span
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import EmbeddingBag, Embedding, LSTM
from config import arg_parse
import json
import itertools as it

args = arg_parse()

class BertDocumentEncoder(nn.Module):
    """ Document encoder for tokens
    """
    def __init__(self, distribute_model=False):
        super().__init__()
        self.distribute_model = distribute_model

        if args.bert_name=='bert-base':
            self.bert, _ = BertModel.from_pretrained("bert-base-cased", output_loading_info=True)
        elif args.bert_name=='bert-large':
            self.bert, _ = BertModel.from_pretrained("bert-large-cased", output_loading_info=True)
        elif args.bert_name=='spanbert-base':
            self.bert, _ = BertModel.from_pretrained("SpanBERT/spanbert-base-cased", output_loading_info=True)
        elif args.bert_name=='spanbert-large':
            self.bert, _ = BertModel.from_pretrained("SpanBERT/spanbert-large-cased", output_loading_info=True)
        elif args.bert_name=='corefbert-base':
            self.bert, _ = BertModel.from_pretrained("nielsr/coref-bert-base", output_loading_info=True)
        elif args.bert_name=='corefbert-large':
            self.bert, _ = BertModel.from_pretrained("nielsr/coref-bert-large", output_loading_info=True)
        elif args.bert_name=='corefroberta-base':
            self.bert, _ = BertModel.from_pretrained("nielsr/coref-roberta-base", output_loading_info=True)
        elif args.bert_name=='corefroberta-large':
            self.bert, _ = BertModel.from_pretrained("nielsr/coref-roberta-large", output_loading_info=True)
        else:
            raise ValueError('A very specific bad thing happened.')
            
        # Dropout
        self.emb_dropout = nn.Dropout(0.3)

        if args.freeze_bert:
            self.freeze()

    def forward(self, doc):
        """ Convert document words to ids, pass through BERT. """
        
        # Tokenize all words, split into sequences of length 128
        # (as per Joshi etal 2019)
        padded_segments = pad_sequence(doc.segments, batch_first=True).long()
        if self.distribute_model:
            padded_segments = padded_segments.cuda(1)
        else:
            padded_segments = padded_segments.cuda(0)

        mask = padded_segments > 0
        # Get hidden states at the last layer of BERT
        embeds = self.bert(padded_segments, attention_mask=mask)[0]
        #print(embeds.shape)
        # Apply embedding dropout
        states = self.emb_dropout(embeds)
        
        # Reshape to a single sequence
        num_segments, seg_len = embeds.shape[0], embeds.shape[1]
        states = states.view(num_segments * seg_len, -1)
        mask = mask.view(-1)
        states = states[mask]

        return states, states

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False

class SentCorelationAnalyzer(nn.Module):
    """
        Computer Sentence Corelations
    """
    def __init__(self, MaxSentLen):
        super().__init__()
        
        self.MaxSentLen = MaxSentLen
        self.sentpair_scorer = LSTM(args.embeds_dim, args.hidden_dim, batch_first=True, bidirectional=True)

    def compute_aoa(self, doc, states):
        sentpair_index = [i for i in it.combinations(range(len(doc.sents)), 2)] + [(i, i) for i in range(len(doc.sents))]
        sentpair_index = sorted(sentpair_index, key=lambda x: (x[0], x[1]))

        sent_length = [bdry[1]-bdry[0]+1 for bdry in doc.sent2subtok_bdry]
        sent_states = list(torch.split(states, sent_length))

        # pad first sentence to fixed length
        sent_states[0] = nn.ConstantPad2d((0,0,0,self.MaxSentLen - sent_length[0]), 0.0)(sent_states[0])
        sent_states_paded = pad_sequence(sent_states, batch_first=True)

        sents_mask = torch.arange(self.MaxSentLen)[None, :] < torch.tensor(sent_length)[:, None]
        sents_mask = torch.where(sents_mask==True, 1.0, 0.0).to(args.device)

        ### compute attention over attention
        documents = sent_states_paded[[sentpair[1] for sentpair in sentpair_index]]
        documents_mask = sents_mask[[sentpair[1] for sentpair in sentpair_index]].unsqueeze(2)
        documents_length = torch.tensor(sent_length)[[sentpair[1] for sentpair in sentpair_index]]
        querys = sent_states_paded[[sentpair[0] for sentpair in sentpair_index]]
        querys_mask = sents_mask[[sentpair[0] for sentpair in sentpair_index]].unsqueeze(2)

        M = torch.bmm(documents, querys.transpose(1, 2))
        M_mask = torch.bmm(documents_mask, querys_mask.transpose(1, 2))

        M_alpha = softmax_mask(M, M_mask, axis=1)
        M_beta = softmax_mask(M, M_mask, axis=2)

        sum_beta = torch.sum(M_beta, dim=1, keepdim=True)
        docs_len = documents_length.unsqueeze(1).unsqueeze(2).expand_as(sum_beta).to(args.device)
        beta = sum_beta / docs_len.float()

        alpha = torch.bmm(M_alpha, beta.transpose(1, 2)).squeeze()
        beta = beta.squeeze()

        self.alpha_metric = torch.zeros(len(doc.sents), len(doc.sents), querys.shape[1])
        self.beta_metric = torch.zeros(len(doc.sents), len(doc.sents), documents.shape[1])
        for id, i in enumerate(sentpair_index):
            self.alpha_metric[i] = alpha[id]
            self.beta_metric[i] = beta[id]
        
        return (self.alpha_metric, self.beta_metric), sentpair_index
        
    def compute_bilstm(self, doc, states):
        sentpair_index = [i for i in it.combinations(range(len(doc.sents)), 2)] + [(i, i) for i in range(len(doc.sents))]
        sentpair_index = sorted(sentpair_index, key=lambda x: (x[0], x[1]))

        sentpair_bdry = [(doc.sent2subtok_bdry[i[0]][0], doc.sent2subtok_bdry[i[0]][1],\
             doc.sent2subtok_bdry[i[1]][0], doc.sent2subtok_bdry[i[1]][1]) for i in sentpair_index]

        token_ranges = []
        for i in sentpair_bdry:
            for j in range(i[0], i[1]+1):
                token_ranges.append(j)
            for j in range(i[2], i[3]+1):
                token_ranges.append(j)

        sentpair_length = [i[1]-i[0]+i[3]-i[2]+2 for i in sentpair_bdry]
        sentpair_embeded = torch.split(states[token_ranges], sentpair_length)

        sentpair_embeded_paded =  pad_sequence(sentpair_embeded, batch_first=True)
        sentpair_embeded_paded_packed = pack_padded_sequence(sentpair_embeded_paded, sentpair_length, batch_first=True, enforce_sorted=False)
        _, (h_output, _) = self.sentpair_scorer(sentpair_embeded_paded_packed)

        h_output = h_output.permute(1,0,2).reshape(h_output.shape[1], -1)

        if args.sent_corelation == 'lstm':
            self.sentpair_metric = torch.zeros(len(doc.sents), len(doc.sents), args.hidden_dim*2)

        # 暴力赋值
        for id, i in enumerate(sentpair_index):
            self.sentpair_metric[i] = h_output[id]

        return self.sentpair_metric

def softmax_mask(input, mask, axis=1, epsilon=1e-12):
    shift, _ = torch.max(input, axis, keepdim=True)
    shift = shift.expand_as(input).to(args.device)

    target_exp = torch.exp(input - shift) * mask

    normalize = torch.sum(target_exp, axis, keepdim=True).expand_as(target_exp)
    softm = target_exp / (normalize + epsilon)

    return softm.to(args.device)

class SentCorelScore(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, input_dim):
        super().__init__()
        self.score_alpha = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, args.sent_corel_dim)
        )

        self.score_beta = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, args.sent_corel_dim)
        )

        self.score_together = nn.Sequential(
            nn.Linear(2*args.sent_corel_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, args.sent_corel_dim)
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(args.sent_corel_dim, 2)
            nn.Softmax()
        )

    def forward(self, x, y):
        """ Output a scalar score for an input x """
        represent_alpha = self.score_alpha(x)
        represent_beta = self.score_beta(y)

        together = torch.cat((represent_alpha, represent_beta), dim=1)

        return self.score_together(together)

class SentCorelModel(nn.Module):
    """
        Super class to compute coreference links between spans
    """

    def __init__(self, MaxSentLen):
        super().__init__()
        
        self.encoder = BertDocumentEncoder()
        
        self.compute_sent_corelation = SentCorelationAnalyzer(MaxSentLen)

        self.corel_sent_score = SentCorelScore(MaxSentLen)

        
    def forward(self, doc):
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, _ = self.encoder(doc)
        
        corel_metrics, sentpair_index = self.compute_sent_corelation.compute_aoa(doc, states)

        # Get pairwise scores for each span combo
        alpha_metric, beta_metric = corel_metrics
        scores = self.corel_sent_score(alpha_metric, beta_metric)

        return scores, sentpair_index