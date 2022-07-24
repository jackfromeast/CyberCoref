from html import entities
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

class Mention_Detection_Score(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embeds_dim, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)

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
        )

    def forward(self, x, y):
        """ Output a scalar score for an input x """
        represent_alpha = self.score_alpha(x)
        represent_beta = self.score_beta(y)

        together = torch.cat((represent_alpha, represent_beta), dim=1)

        return self.score_together(together)

class Distance(nn.Module):
    """ Learned, continuous representations for: distance
    between spans
    """

    bins = torch.LongTensor([1,2,3,4,8,16,32,64,128,256,384]).to(args.device)

    def __init__(self, distance_dim=20):
        super().__init__()

        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, distance_dim),
            nn.Dropout(0.3)
        )

    def forward(self, *args):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def stoi(self, lengths):
        """ Find which bin a number falls into """
        return torch.sum(lengths.unsqueeze(1) > self.bins, dim=1)


class Width(nn.Module):
    """ Learned, continuous representations for: span width
    """
    
    def __init__(self, L, width_dim=20):
        super().__init__()

        self.dim = width_dim
        self.embeds = nn.Sequential(
            nn.Embedding(L, width_dim),
        )

    def forward(self, widths):
        """ Embedding table lookup """
        return self.embeds(widths)

class Type(nn.Module):
    """ Learned, continuous representations for: span types
    """
    
    def __init__(self, embed_dim=64):
        super().__init__()

        self.dim = embed_dim
        self.embeds = nn.Sequential(
            nn.Embedding(30, embed_dim),
        )

    def forward(self, types):
        """ Embedding table lookup """
        return self.embeds(types)


class BertDocumentEncoder(nn.Module):
    """ Document encoder for tokens
    """
    def __init__(self, distribute_model=False, TypePredictor=None):
        super().__init__()
        self.distribute_model = distribute_model

        if args.tp_all_in_one:
            self.bert = TypePredictor.typePredictor.encoder.bert
        elif args.bert_name=='bert-base':
            self.bert, _ = BertModel.from_pretrained("bert-base-cased", output_loading_info=True)
        elif args.bert_name=='bert-large':
            self.bert, _ = BertModel.from_pretrained("bert-large-cased", output_loading_info=True)
        elif args.bert_name=='spanbert-base':
            # self.bert, _ = BertModel.from_pretrained("SpanBERT/spanbert-base-cased", output_loading_info=True)
            self.bert, _ = BertModel.from_pretrained("./BERTs/spanbert-base-cased", output_loading_info=True)
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

        padded_segments = padded_segments.to(args.device)

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


class POSDeprelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        with open(args.dict_path+'deprel_dict.json', 'r') as f:
            deprel_tags = json.load(f)
        with open(args.dict_path+'pos_dict.json', 'r') as f:
            pos_tags = json.load(f)
        self.deprel2idx = {deprel_tag: idx+1 for idx, deprel_tag in enumerate(deprel_tags.values())}
        self.pos2idx = {pos_tag: idx+1 for idx, pos_tag in enumerate(pos_tags.values())}

        if args.pd_solution in ['sum', 'mean']:
            self.pos_embed = EmbeddingBag(num_embeddings=len(pos_tags)+1, embedding_dim=args.pos_dim, padding_idx=0, mode=args.pd_solution)
            self.deprel_embed = EmbeddingBag(num_embeddings=len(deprel_tags)+1, embedding_dim=args.deprel_dim, padding_idx=0, mode=args.pd_solution)
        else:
            self.pos_embed = Embedding(num_embeddings=len(pos_tags)+1, embedding_dim=args.pos_dim, padding_idx=0)
            self.deprel_embed = Embedding(num_embeddings=len(deprel_tags)+1, embedding_dim=args.deprel_dim, padding_idx=0)
        
            if args.pd_solution == 'lstm':
                self.pos_LSTM = LSTM(args.pos_dim, args.pos_dim, num_layers=1, batch_first=True)
                self.deprel_LSTM = LSTM(args.deprel_dim, args.deprel_dim, num_layers=1, batch_first=True)
            elif args.pd_solution == 'attn':
                self.v = torch.nn.Parameter(torch.FloatTensor(args.pos_dim).uniform_(-0.1, 0.1))
                self.W_1 = torch.nn.Linear(args.pos_dim, args.pos_dim)
                self.W_2 = torch.nn.Linear(args.deprel_dim, args.deprel_dim)
    
    def encode(self, start_words, end_words, doc):
        pos = []
        deprel = []

        if args.pd_solution in ['sum', 'mean']:
            for st, ed in zip(start_words, end_words):
                pos.append(torch.tensor([self.pos2idx[tag] for tag in doc.pos[st: ed+1]], dtype=torch.long))
                deprel.append(torch.tensor([self.deprel2idx[tag] for tag in doc.deprel[st: ed+1]], dtype=torch.long))
            padded_pos = pad_sequence(pos, padding_value=0, batch_first=True).to(args.device)
            padded_deprel = pad_sequence(deprel, padding_value=0, batch_first=True).to(args.device)

            embedded_pos = self.pos_embed(padded_pos)
            embedded_deprel = self.deprel_embed(padded_deprel)

            return embedded_pos, embedded_deprel

        elif args.pd_solution == 'lstm':
            seq_lens = []
            for st, ed in zip(start_words, end_words):
                pos.append(torch.tensor([self.pos2idx[tag] for tag in doc.pos[st: ed+1]]))
                deprel.append(torch.tensor([self.deprel2idx[tag] for tag in doc.deprel[st: ed+1]]))
                seq_lens.append(ed-st+1)

            padded_pos = pad_sequence(pos, padding_value=0, batch_first=True).to(args.device)
            padded_deprel = pad_sequence(deprel, padding_value=0, batch_first=True).to(args.device)

            embedded_pos = self.pos_embed(padded_pos)
            embedded_deprel = self.deprel_embed(padded_deprel)

            packed_pos = pack_padded_sequence(embedded_pos, seq_lens, batch_first=True, enforce_sorted=False)
            packed_deprel = pack_padded_sequence(embedded_deprel, seq_lens, batch_first=True, enforce_sorted=False)

            # 取LSTM最后非pad的时间步的Hidden State
            output_pos, (h_pos, _) = self.pos_LSTM(packed_pos)
            output_deprel, (h_rel, _) = self.deprel_LSTM(packed_deprel)

            h_pos = torch.squeeze(h_pos)
            h_rel = torch.squeeze(h_rel)

            return h_pos, h_rel
        
        elif args.pd_solution == 'attn':
            for st, ed in zip(start_words, end_words):
                cur_pos_tags = []
                cur_deprel_tags = []
                for idx, tag in enumerate(doc.pos[st: ed+1], st):
                    cur_pos_tags += [self.pos2idx[tag]] * len(doc.word2subtok[idx])
                for idx, tag in enumerate(doc.deprel[st: ed+1], st):
                    cur_deprel_tags += [self.deprel2idx[tag]] * len(doc.word2subtok[idx])
                pos.append(torch.tensor(cur_pos_tags))
                deprel.append(torch.tensor(cur_deprel_tags))

            padded_pos = pad_sequence(pos, padding_value=0, batch_first=True).to(args.device)
            padded_deprel = pad_sequence(deprel, padding_value=0, batch_first=True).to(args.device)
            mask = torch.where(padded_pos==0, False, True)

            embedded_pos = self.pos_embed(padded_pos)
            embedded_deprel = self.deprel_embed(padded_deprel)

            weights = self.W_1(embedded_pos) + self.W_2(embedded_deprel)
            weights = torch.tanh(weights) @ self.v

            return  F.softmax(weights.masked_fill(mask==False, -1e9), dim=1)


class typePred():
    def __init__(self, typePredictor):
        self.predictor = typePredictor
    
    def _batched_input(self):
        paded_sents = pad_sequence(self.new_all_tcs, batch_first=True).to(args.device)
        span_pos = torch.tensor(self.new_all_tnb).to(args.device)
    
        self.batched_sents = torch.split(paded_sents, 64)
        self.batched_span_pos = torch.split(span_pos, 64)

    def predict(self, doc, sent_ids, start_toks, end_toks):
        if args.tp_solution == 'gold':
            self.result = self.gold_types(doc, start_toks, end_toks)
            return self.result
        elif args.insertTag:
            self.tagged_preprocess(doc, sent_ids, start_toks, end_toks)

            self._batched_input()

            self.predictor.eval()

            self.result = torch.tensor([], dtype=torch.float).to(args.device)
            for i in range(len(self.batched_sents)):
                result = self.predictor.forward(self.batched_sents[i], self.batched_span_pos[i])

                self.result = torch.cat([self.result, result], dim=0)

            return self.result
        else:
            self.preprocess(doc, sent_ids, start_toks, end_toks)

            paded_sents = pad_sequence(self.doc_sents, batch_first=True).to(args.device)
            tokens_bdry = torch.tensor(self.tokens_bdry).to(args.device)

            sent_embeds, _ = self.predictor.typePredictor.encoder(paded_sents)

            token_cor_sent_embeds = sent_embeds[self.sent_ids]

            batched_sents = torch.split(token_cor_sent_embeds, 128)
            batched_span_pos = torch.split(tokens_bdry, 128)

            self.result = torch.tensor([], dtype=torch.float).to(args.device)
            for i in range(len(batched_sents)):
                result = self.predictor.typePredictor.scorer(batched_sents[i], batched_span_pos[i])

                self.result = torch.cat([self.result, result], dim=0)

            return self.result

    def tagged_preprocess(self, doc, sent_ids, start_toks, end_toks):
        # 去除头尾的101和102 对每条句子重新添加
        self.new_all_tnb, self.new_all_tcs = [], []

        sents_bdry = [[doc.sent2subtok_bdry[0][0], doc.sent2subtok_bdry[0][1]-1]]
        sents_bdry += [[bdry[0]-1, bdry[1]-1]for bdry in doc.sent2subtok_bdry[1:-1]]
        sents_bdry += [[doc.sent2subtok_bdry[-1][0]-1, doc.sent2subtok_bdry[-1][1]-2]]

        tokens_new_bdry = [[s, e-1] for s,e in zip(start_toks, end_toks) if s==0]
        tokens_new_bdry += [[s-1, e-1] for s,e in zip(start_toks, end_toks) if s!=0 and e!=doc.sent2subtok_bdry[-1][1]]
        tokens_new_bdry += [[s-1, e-2] for s,e in zip(start_toks, end_toks) if e==doc.sent2subtok_bdry[-1][1]]

        token_cor_sents = [doc.bert_tokens[sents_bdry[sent_id][0]:sents_bdry[sent_id][1]+1] for sent_id in sent_ids]
        for tnb, tcs, sent_id in zip(tokens_new_bdry, token_cor_sents, sent_ids):
            new_tcs = [101] + tcs[:tnb[0]] + [28996] + tcs[tnb[0]:tnb[1]+1] + [28997] + tcs[tnb[1]+1:] +[102]
            new_tnb = [tnb[0]-sents_bdry[sent_id][0]+2, tnb[1]-sents_bdry[sent_id][0]+2] # 处理语句的offset

            self.new_all_tnb.append(new_tnb)
            self.new_all_tcs.append(torch.tensor(new_tcs))
        
    def preprocess(self, doc, sent_ids, start_toks, end_toks):
        # 去除每个segment头尾的101和102 对每条句子重新添加

        segment_lenth = [len(segment) for segment in doc.segments]
        segment_starts = [sum(segment_lenth[:i]) for i in range(len(doc.segments))]
        segment_ends = [sum(segment_lenth[:i+1])-1 for i in range(len(doc.segments))]

        sents_bdry = []
        for bdry in doc.sent2subtok_bdry:
            cur_seg_num = 0
            for end in segment_ends:
                if bdry[1] <= end:
                    break
                else:
                    cur_seg_num += 1 
            
            if bdry[0] in segment_starts:
                sents_bdry.append([bdry[0]-2*cur_seg_num, bdry[1]-1-2*cur_seg_num])
            if bdry[0] not in segment_starts and bdry[1] not in segment_ends:
                sents_bdry.append([bdry[0]-1-2*cur_seg_num, bdry[1]-1-2*cur_seg_num])
            if bdry[1] in segment_ends:
                sents_bdry.append([bdry[0]-1-2*cur_seg_num, bdry[1]-2-2*cur_seg_num])

        tokens_new_bdry = []
        for s, e in zip(start_toks, end_toks):
            cur_seg_num = 0
            for end in segment_ends:
                if e <= end:
                    break
                else:
                    cur_seg_num += 1 
            if s in segment_starts:
                tokens_new_bdry.append([s-2*cur_seg_num, e-1-2*cur_seg_num])
            elif s not in segment_starts and e not in segment_ends:
                tokens_new_bdry.append([s-1-2*cur_seg_num, e-1-2*cur_seg_num])
            elif e in segment_ends:
                tokens_new_bdry.append([s-1-2*cur_seg_num, e-2-2*cur_seg_num])

        self.doc_sents = [torch.tensor([101]+doc.bert_tokens[bdry[0]:bdry[1]+1]+[102]) for bdry in sents_bdry]
        self.tokens_bdry = [[span_pos[0]+1-sents_bdry[sent_id][0], span_pos[1]+1-sents_bdry[sent_id][0]] for span_pos,sent_id in zip(tokens_new_bdry, sent_ids)] # added [101] and [102]
        self.sent_ids = sent_ids

        ### check
        for bdry, sent_id in zip(self.tokens_bdry, sent_ids):
            assert bdry[0] < self.doc_sents[sent_id].size(0) and bdry[1] < self.doc_sents[sent_id].size(0), 'Error!'

            assert bdry[0] >= 0 and bdry[1] >= 0, 'Error!'

    def gold_types(self, doc, start_toks, end_toks):
        with open('./Dataset/others/type2label.json', 'r') as fs:
            type2label = json.load(fs)
        
        entity_types = {}
        for entity in doc.entities:
            for coref in doc.corefs:
                try:
                    if coref['span'][0]==entity[5][0] and coref['span'][-1]==entity[5][-1]:
                        entity_types[(doc.word2subtok[coref['span'][0]][0], doc.word2subtok[coref['span'][-1]][-1])]=entity[1]
                except(IndexError):
                    continue
        # entity_types = {(doc.word2subtok[coref['span'][0]][0], doc.word2subtok[coref['span'][-1]][-1]):entity[1] for entity in doc.entities for coref in doc.corefs if coref['span'][0]==entity[5][0] and coref['span'][-1]==entity[5][-1]}

        result = []
        for s,e in zip(start_toks, end_toks):
            if (s,e) in entity_types.keys():
                result.append(type2label[entity_types[(s,e)]])
            else:
                result.append(type2label['None'])

        result = F.one_hot(torch.tensor(result)).float().to(args.device)
        return result
      
class MentionScore(nn.Module):
    """ Mention scoring module
    """
    
    def __init__(self, gi_dim, attn_dim, distance_dim, L, distribute_model, typePredictor):
        super().__init__()

        self.attention = Score(attn_dim)
        self.width = Width(L, distance_dim)
        # self.score = Mention_Detection_Score(gi_dim)
        self.score = Score(gi_dim)
        self.typePred = typePred(typePredictor)
        self.types = Type(args.type_dim)
        self.L = L
        self.distribute_model = distribute_model

        if args.pd_solution != None:
            self.pos_deprel_encoder = POSDeprelEncoder()
        
    def forward(self, states, embeds, doc):
        
        #Compute unary mention score for each span
        word2tokens = doc.word2subtok
        
        start_words, end_words, start_toks, end_toks, \
                tok_ranges, word_widths, tok_widths, sent_ids = compute_idx_spans_for_bert(doc.sents, self.L, word2tokens)
        
        # Compute first part of attention over span states (alpha_t)
        attns = self.attention(states)
        # Regroup attn values, embeds into span representations
        assert sum(tok_widths) == len(tok_ranges)
        span_attns, span_embeds = torch.split(attns[tok_ranges], tok_widths), \
                                     torch.split(states[tok_ranges], tok_widths)
        
        # Pad and stack span attention values, span embeddings for batching
        padded_attns = pad_and_stack(span_attns, value=-1e10)
        padded_embeds = pad_and_stack(span_embeds, )
        # Weight attention values using softmax
        attn_weights = F.softmax(padded_attns, dim=1)

        # Compute self-attention over embeddings (x_hat)
        attn_embeds = torch.sum(padded_embeds * attn_weights, dim=1)

        ## Compute other features like widths
        widths = self.width(torch.tensor(word_widths).to(args.device))

        # Compute type information from typePredictor with loaded weights
        type_preds = self.typePred.predict(doc, sent_ids, start_toks, end_toks)

        type_preds = nn.functional.softmax(type_preds, dim=1) # (spans_num, 30)
        type_embeds = self.types(torch.argmax(type_preds, dim=1)) # (spans_num, type_dim)

        ## Compute POS and Deprel vectors
        if args.pd_solution in ['sum', 'mean', 'lstm']:
            embedded_pos, embedded_deprel = self.pos_deprel_encoder.encode(start_words, end_words, doc)
            g_i_mention = torch.cat((states[start_toks], states[end_toks], attn_embeds, embedded_pos, embedded_deprel, widths), dim=1)
        elif args.pd_solution == 'attn':
            pd_attn_weights = self.pos_deprel_encoder.encode(start_words, end_words, doc)
            pd_attn_embeds = torch.sum(padded_embeds * torch.unsqueeze(pd_attn_weights, -1), dim=1)
            g_i_mention = torch.cat((states[start_toks], states[end_toks], attn_embeds, pd_attn_embeds, widths), dim=1)
        elif args.tp_solution != 'None':
            g_i_mention = torch.cat((states[start_toks], states[end_toks], attn_embeds, widths, type_preds, type_embeds), dim=1)
        else:
            g_i_mention = torch.cat((states[start_toks], states[end_toks], attn_embeds, widths), dim=1)

        # Cat it all together to get g_i, our span representation
        if args.mention_coref_gi_split:
            g_i_coref = torch.cat((states[start_toks], states[end_toks], attn_embeds, widths), dim=1)
        else:
            g_i_coref = g_i_mention

        # Compute each span's unary mention score
        mention_scores = self.score(g_i_mention)
        # mention_scores = torch.unsqueeze(span_scores[:,1] - span_scores[:,0], -1)
        
        ### Expriments for coref resolution with all gold mentions
        # mention_scores = []
        # gold_spans = [coref['span'] for coref in doc.corefs]
        # for start_word, end_word in zip(start_words, end_words):
        #     if (start_word, end_word) in gold_spans:
        #         mention_scores.append([1.0])
        #     else:
        #         mention_scores.append([-1.0])
        # mention_scores = torch.tensor(mention_scores, dtype=torch.float32).to(args.device)

        # Prune down to LAMBDA*len(doc) spans
        indices_sorted = prune_bert(mention_scores, start_words, end_words, doc)
        
        # Create span objects here
        selected_spans = [Span(i1=start_words[idx], 
                      i2=end_words[idx], 
                      id=idx,
                      si=mention_scores[idx],
                      sent_id = sent_ids[idx],
                      type = type_preds[idx],
                      type_embed = type_embeds[idx],
                      content=doc.tokens[start_words[idx]:end_words[idx]+1])
                 for idx in indices_sorted]
        
        all_spans = [Span(i1=start_words[idx], 
                      i2=end_words[idx], 
                      id=idx,
                      si=mention_scores[idx],
                      sent_id = sent_ids[idx],
                      type = type_preds[idx],
                      type_embed = type_embeds[idx],
                      content=doc.tokens[start_words[idx]:end_words[idx]+1])
                 for idx in range(len(start_words))]
        
        return selected_spans, mention_scores, all_spans, mention_scores, g_i_coref
                
class HigherOrderScore(nn.Module):
    """ Coreference pair scoring module
    """
    def __init__(self, gij_dim, gi_dim, distance_dim, K, N, MaxSentLen):
        super().__init__()
        self.distance = Distance(distance_dim)
        self.distance_coarse = Distance(distance_dim)

        self.corel_score = SentCorelScore(MaxSentLen)

        self.score = Score(gij_dim)
        
        self.coarse_W = nn.Linear(gi_dim, gi_dim, bias=False)
        self.dropout = nn.Dropout(0.3)
                    
        self.W_f = nn.Linear(gi_dim*2, gi_dim)
        self.distance_proj = nn.Linear(distance_dim, 1)
        
        self.bilin = nn.Bilinear(gi_dim, gi_dim, 1)
        
        self.K = K
        self.N = N
        
        
    def forward(self, spans, g_i, mention_scores, corel_metrics=None):
        """ Compute pairwise score for spans and their up to K antecedents
        """
        # ================================================================
        # Second stage: coarse pruning
        # Get the antecedent IDs for current spans
        mention_ids, start_indices, end_indices = zip(*[(span.id, span.i1, span.i2)
                                                        for span in spans])
        
        mention_ids = torch.tensor(mention_ids).to(args.device)
        start_indices = torch.tensor(start_indices).to(args.device)
        end_indices = torch.tensor(end_indices).to(args.device)
        
        i_g = torch.index_select(g_i, 0, mention_ids)
        s_i = torch.index_select(mention_scores, 0, mention_ids)
        
        k = mention_ids.shape[0]
        top_span_range = torch.arange(k)
        antecedent_offsets = (top_span_range.unsqueeze(1) - top_span_range.unsqueeze(0)).to(args.device)
        antecedent_mask = antecedent_offsets >= 1
        
        antecedent_scores = torch.mm(self.dropout(
                                         self.coarse_W(i_g)
                                     ), 
                                     self.dropout(
                                         i_g.transpose(0, 1)
                                     ))
        
        distances = end_indices.unsqueeze(1) - start_indices.unsqueeze(0)
        distances = self.distance_proj(self.distance_coarse(distances.view(k * k))).view(k, k)

        antecedent_scores += s_i + s_i.transpose(0, 1) + torch.log(antecedent_mask.float())
        
        
        best_scores, best_indices = torch.topk(antecedent_scores, 
                                  k=min(self.K, antecedent_scores.shape[0]),
                                  sorted=False)
        all_best_scores = []
        
        spans[0] = attr.evolve(spans[0], yi=[])
        spans[0] = attr.evolve(spans[0], yi_idx=[])
        
        for i, span in enumerate(spans[1:], 1):
            
            yi, yi_idx = zip(*[(spans[idx], 
                               ((spans[idx].i1, spans[idx].i2), (span.i1, span.i2))) 
                               for idx in best_indices[i][:i]])
            
            spans[i] = attr.evolve(spans[i], yi=yi)
            spans[i] = attr.evolve(spans[i], yi_idx=yi_idx)
            all_best_scores.append(best_scores[i, :i])
            
        s_ij_c = torch.cat(all_best_scores, dim=0).unsqueeze(1)
        
        # ===================================================================
        # Third stage: second-order inference
        # Extract raw features        
        mention_ids, antecedent_ids, \
            distances, sentpair_ids, \
                mention_types, antecedent_types, \
                    mention_type_embeds, antecedent_type_embeds = zip(*[(i.id, j.id,
                                                i.i2-j.i1, (i.sent_id, j.sent_id),
                                                i.type, j.type, i.type_embed, j.type_embed)
                                             for i in spans
                                             for j in i.yi])
        
        # Embed them
        distances = torch.tensor(distances).to(args.device)
        mention_types = torch.stack(mention_types, dim=0).to(args.device)
        antecedent_types = torch.stack(antecedent_types, dim=0).to(args.device)
        mention_type_embeds = torch.stack(mention_type_embeds, dim=0).to(args.device)
        antecedent_type_embeds = torch.stack(antecedent_type_embeds, dim=0).to(args.device)
        
        type_similarity =  F.cosine_similarity(mention_types, antecedent_types).unsqueeze(-1)
        phi = torch.cat((self.distance(distances), \
            type_similarity, mention_type_embeds, antecedent_type_embeds), dim=1)
        
        # For indexing a tensor efficiently
        mention_ids = torch.tensor(mention_ids).to(args.device)
        antecedent_ids = torch.tensor(antecedent_ids).to(args.device)
        
        # Get antecedent indexes for each span (first span has no antecedents)
        antecedent_idx = [len(s.yi) for s in spans[1:]]
        unique_mention_ids = torch.tensor([span.id for span in spans[1:]]).to(args.device)
        epsilon = torch.zeros(unique_mention_ids.shape[0], 1).to(args.device)
        
        for step in range(self.N):

            # Extract their span representations from the g_i matrix
            i_g = torch.index_select(g_i, 0, mention_ids)
            j_g = torch.index_select(g_i, 0, antecedent_ids)

            if args.sent_corelation == 'bilstm':
                sent_corel = torch.tensor([corel_metrics[sentpair_id].tolist() for sentpair_id in sentpair_ids])
                pairs = torch.cat((i_g, j_g, i_g*j_g, sent_corel, phi), dim=1)

            elif args.sent_corelation == 'dattn':
                alpha_metric, beta_metric = corel_metrics
                span_alpha = torch.tensor([alpha_metric[sentpair_id].tolist() for sentpair_id in sentpair_ids]).to(args.device)
                span_beta = torch.tensor([beta_metric[sentpair_id].tolist() for sentpair_id in sentpair_ids]).to(args.device)

                corel_embeded = self.corel_score(span_alpha, span_beta)

                pairs = torch.cat((i_g, j_g, i_g*j_g, corel_embeded, phi), dim=1)

            else:
                pairs = torch.cat((i_g, j_g, i_g*j_g, phi), dim=1)

            # Score pairs of spans for coreference link
            s_ij_a = self.score(pairs)
            # Compute pairwise scores for coreference links between each mention and
            # its antecedents
            coref_scores = s_ij_a + s_ij_c
            # Split coref scores so each list entry are scores for its antecedents, only.
            # (NOTE that first index is a special case for torch.split, so we handle it here)
            split_scores = [torch.tensor([[0.0]]).to(args.device)] \
                             + list(torch.split(coref_scores, antecedent_idx, dim=0))
            
            if step == self.N - 1:
                break

            # Compute probabilities for antecedents
            p_yi = pad_and_stack(pad_and_stack(split_scores[1:], 
                                          value=-1e10))
            p_yi = F.softmax(torch.cat([epsilon.unsqueeze(1), p_yi], dim=1), dim=1)
            
            mentions = g_i[unique_mention_ids]
            
            # Mention vector updates from antecedets:
            a_n = pad_and_stack(torch.split(j_g, antecedent_idx, dim=0))
            a_n = torch.sum(torch.cat([mentions.unsqueeze(1), a_n], dim=1) * p_yi, dim=1)
            
            f_n = torch.sigmoid(self.W_f(torch.cat((mentions, a_n), dim=1)))
            g_i = g_i.clone()
            g_i[unique_mention_ids] = f_n * mentions + (1 - f_n) * a_n
            
        scores = pad_and_stack(split_scores, value=-1e10).squeeze(2)
        scores = torch.cat([torch.zeros(epsilon.shape[0]+1, 1).to(args.device), scores], dim=1)
        return spans, scores


class SentCorelationAnalyzer(nn.Module):
    """
        Computer Sentence Corelations
    """
    def __init__(self, MaxSentLen):
        super().__init__()
        
        self.MaxSentLen = MaxSentLen
        self.sentpair_scorer = LSTM(args.embeds_dim, args.hidden_dim, batch_first=True, bidirectional=True)

    @staticmethod
    def batched_cos_sim(A, B, dim=-1, eps=1e-8):
    #   numerator = A @ B.T
      numerator = torch.bmm(A, B.transpose(1, 2))
      A_l2 = torch.mul(A, A).sum(axis=dim)
      B_l2 = torch.mul(B, B).sum(axis=dim)
      denominator = torch.max(torch.sqrt(torch.einsum('bi,bj->bij', (A_l2, B_l2))), torch.tensor(eps))
      return torch.div(numerator, denominator)

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

        # M = torch.bmm(documents, querys.transpose(1, 2))
        M = self.batched_cos_sim(documents, querys)
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
        
        return (self.alpha_metric, self.beta_metric)

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

class CyberCorefScorer(nn.Module):
    """
        Super class to compute coreference links between spans
    """

    def __init__(self, MaxSentLen,
                       typePredictor,
                       distribute_model=args.distribute_model,
                       attn_dim = args.atten_dim,
                       embeds_dim = args.embeds_dim,
                       distance_dim=args.distance_dim,
                       pos_dim=args.pos_dim,
                       deprel_dim=args.deprel_dim):
        super().__init__()
        assert embeds_dim == 768, 'For bert-based model, embeds_dim should match the size with pre-trained BERT-based Model.'
        
        if args.tp_all_in_one:
            self.encoder = BertDocumentEncoder(distribute_model, typePredictor)
        else:
            self.encoder = BertDocumentEncoder(distribute_model)

        self.typePredictor = typePredictor

        # Forward and backward passes, avg'd attn over embeddings, span width
        if args.pd_solution in ['sum', 'mean', 'lstm']:
            gi_mention_dim = attn_dim*2 + embeds_dim + pos_dim + deprel_dim + distance_dim   
        elif args.pd_solution == 'attn':
            gi_mention_dim = attn_dim*2 + embeds_dim*2 + distance_dim
        else:
            gi_mention_dim = attn_dim*2 + embeds_dim + distance_dim
        
        if args.tp_solution != 'None':
            gi_mention_dim += 30 + args.type_dim

        if args.mention_coref_gi_split:
            gi_coref_dim = attn_dim*2 + embeds_dim + distance_dim
        else:
            gi_coref_dim = gi_mention_dim


        # gi, gj, gi*gj, distance between gi and gj
        if args.sent_corelation == 'dattn':
            gij_dim = gi_coref_dim*3 + distance_dim + args.sent_corel_dim
        else:
            gij_dim = gi_coref_dim*3 + distance_dim
        
        if args.tp_solution != 'None':
            gij_dim += 1 + 2*args.type_dim
            
        self.score_spans = MentionScore(gi_mention_dim, attn_dim, distance_dim, L=args.max_span_length, 
                                        distribute_model=distribute_model, typePredictor=self.typePredictor)
        
        self.compute_sent_corelation = SentCorelationAnalyzer(MaxSentLen)

        self.score_pairs = HigherOrderScore(gij_dim, 
                                            gi_coref_dim, distance_dim, K=args.top_K, N=args.higer_order_N, MaxSentLen=MaxSentLen)

    def forward(self, doc):
        """
            Encode document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores..
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.encoder(doc)
        
        # Get mention scores for each span, prune
        selected_spans, mention_scores, all_spans, all_span_scores, g_i = self.score_spans(states, embeds, doc)
        # If the document is too short
        if len(selected_spans) <= 2:
            return None, None

        # Get pairwise scores for each span combo
        if args.sent_corelation == 'dattn':
            corel_metrics = self.compute_sent_corelation.compute_aoa(doc, states)
            selected_spans, coref_scores = self.score_pairs(selected_spans, g_i, mention_scores, corel_metrics)
        else:
            selected_spans, coref_scores = self.score_pairs(selected_spans, g_i, mention_scores)

        return selected_spans, coref_scores, all_spans, all_span_scores
