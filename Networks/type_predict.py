import sys 
sys.path.append("..") 
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
import attr
from utils import *
from dataLoader import Span

from config import arg_parse
args = arg_parse()

class Width(nn.Module):
    """ Learned, continuous representations for: span width
    """

    bins = torch.LongTensor([1,2,3,4,5,6,7,8,12,16,20,24,32,64,128]).to(args.device)

    def __init__(self, width_dim=20):
        super().__init__()

        self.dim = width_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, width_dim),
            nn.Dropout(0.3)
        )

    def forward(self, *args):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def stoi(self, lengths):
        """ Find which bin a number falls into """
        return torch.sum(lengths.unsqueeze(1) > self.bins, dim=1)

class Score(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embeds_dim, 1000),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(1000, 1)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)

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
            # self.bert, _ = BertModel.from_pretrained("SpanBERT/spanbert-base-cased", output_loading_info=True)
            self.bert, _ = BertModel.from_pretrained("./BERTs/spanbert-base-cased", output_loading_info=True)
            # bert_tokenizer = BertTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
            bert_tokenizer = BertTokenizer.from_pretrained("./BERTs/spanbert-base-cased")
            bert_tokenizer.add_tokens("<SST>", special_tokens=True)
            bert_tokenizer.add_tokens("<SND>", special_tokens=True)
            self.bert.resize_token_embeddings(len(bert_tokenizer))
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

    def forward(self, batched_sents):
        """ Convert document words to ids, pass through BERT. """
        
        # Tokenize all words, split into sequences of length 128
        # (as per Joshi etal 2019)
        # padded_segments = pad_sequence(doc.segments, batch_first=True).long()
        padded_segments = batched_sents
        padded_segments = padded_segments.to(args.device)

        mask = padded_segments > 0
        # Get hidden states at the last layer of BERT
        embeds = self.bert(padded_segments, attention_mask=mask)[0]
        # print(embeds.shape)
        # Apply embedding dropout
        states = self.emb_dropout(embeds)
        # states = states[mask]
        
        return states, states

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False

class TypeScorer(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, scorer_input_dim):
        super().__init__()
        
        if args.tp_solution == 'tagged-token-md' or args.tp_solution == 'without-tag-md':
            self.attention = Score(args.atten_dim)
            self.width = Width(width_dim=args.width_dim)

        if scorer_input_dim > 1024:
            self.score = nn.Sequential(
                nn.Linear(scorer_input_dim, 1024),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 30),
            )
        else:
            self.score = nn.Sequential(
                nn.Linear(scorer_input_dim, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 30),
            )

    def forward(self, embeds, spans):
        if args.tp_solution == 'tag':
            tag_pos = torch.tensor([[span[0]-1, span[1]+1] for span in spans], dtype=torch.long).to(args.device)
            # reshape for gather
            tag_pos = tag_pos.unsqueeze(-1).repeat(1, 1, 768)
            tag_embeds = torch.gather(embeds, 1, tag_pos)

            # 拼接<SST>和<SED>对应嵌入向量并降维
            tag_embeds = tag_embeds.view(tag_embeds.size(0), -1)

            return self.score(tag_embeds)
        
        # use tagged sentence, mean pooling the spans
        elif args.tp_solution == 'tagged-mean' or args.tp_solution == 'without-tag-mean':
            span_pos = torch.tensor([[span[0], span[1]] for span in spans], dtype=torch.long).to(args.device)
            span_widths = torch.tensor([pos[1]-pos[0]+1 for pos in span_pos], dtype=torch.long).to(args.device)

            cols = torch.LongTensor(range(embeds.size(1))).repeat(embeds.size(0), 1).to(args.device)
            beg = span_pos[:,0].unsqueeze (1).repeat (1, embeds.size(1))
            end = span_pos[:,1].unsqueeze (1).repeat (1, embeds.size(1))
            mask = cols.ge (beg) & cols.lt (end)

            mask = mask.unsqueeze(-1).repeat(1, 1, 768)
            span_embeds = embeds.masked_fill_(mask == False, 0.0)

            span_embeds = torch.div(torch.sum(span_embeds, dim=1), span_widths.unsqueeze(-1).repeat(1, 768))

            return self.score(span_embeds)
        
        # mention dectection representation
        elif args.tp_solution == 'tagged-token-md' or args.tp_solution == 'without-tag-md':
            span_pos = torch.tensor([[span[0], span[1]] for span in spans], dtype=torch.long).to(args.device)
            span_widths = torch.tensor([pos[1]-pos[0]+1 for pos in span_pos], dtype=torch.long).to(args.device)
            span_pos_repeated = span_pos.unsqueeze(-1).repeat(1, 1, 768)
            span_se_embeds = torch.gather(embeds, 1, span_pos_repeated).view(embeds.size(0), -1) # (batch_size, 2* embeds_dim)

            attns = self.attention(embeds).squeeze(-1)
            attn_embeds = torch.zeros(embeds.size(0), embeds.size(2)).to(args.device) # (batch_size, embeds_dim)
            # TODO: batched compute
            for i in range(embeds.size(0)):
                span_embeds = embeds[i, span_pos[i][0]:span_pos[i][1]+1, :] # 2*768
                attn_weights = nn.functional.softmax(attns[i, span_pos[i][0]:span_pos[i][1]+1], dim=0).unsqueeze(-1) # 2*1

                attn_embed = torch.sum(span_embeds * attn_weights, dim=0) # (embeds_dim, )

                attn_embeds[i] = attn_embed

            widths =  self.width(span_widths).to(args.device)

            span_embeds = torch.cat((span_se_embeds, attn_embeds, widths), dim=1)

            return self.score(span_embeds)


class typePredictor(nn.Module):
    """
        Super class to compute span's corresponding type
    """

    def __init__(self, distribute_model=args.distribute_model,
                       attn_dim = args.atten_dim,
                       embeds_dim = args.embeds_dim,
                       width_dim=args.width_dim,
                       genre_dim=args.genre_dim,
                       speaker_dim=args.speaker_dim):
        super().__init__()
        assert embeds_dim == 768, 'For bert-based model, embeds_dim should match the size with pre-trained BERT-based Model.'
        
        if args.tp_solution == 'tag':
            scorer_input_dim = embeds_dim*2
        elif args.tp_solution == 'tagged-mean' or args.tp_solution == 'without-tag-mean':
            scorer_input_dim = embeds_dim
        elif args.tp_solution == 'tagged-token-md' or args.tp_solution == 'without-tag-md':
            scorer_input_dim = embeds_dim*2 + attn_dim + width_dim

        self.encoder = BertDocumentEncoder(distribute_model)

        self.scorer = TypeScorer(scorer_input_dim)


    def forward(self, batched_sents, spans):
        """
            Encode document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores..
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.encoder(batched_sents)
        
        pred_types = self.scorer(embeds, spans)
       
        return pred_types
