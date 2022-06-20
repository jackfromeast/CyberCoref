import sys 
sys.path.append("..") 
import torch
import torch.nn as nn
from transformers import BertModel
import attr
from utils import *
from dataLoader import Span

from config import arg_parse
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


class Distance(nn.Module):
    """ Learned, continuous representations for: distance
    between spans
    """

    bins = torch.cuda.LongTensor([1,2,3,4,8,16,32,64,128,256,384])

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
    

class Genre(nn.Module):
    """ Learned continuous representations for genre. Zeros if genre unknown.
    """

    genres = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
    _stoi = {genre: idx+1 for idx, genre in enumerate(genres)}

    def __init__(self, genre_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(len(self.genres)+1, genre_dim, padding_idx=0),
            nn.Dropout(0.3)#0.2
        )

    def forward(self, labels):
        """ Embedding table lookup """
        return self.embeds(self.stoi(labels))

    def stoi(self, labels):
        """ Locate embedding id for genre """
        indexes = [self._stoi.get(gen) for gen in labels]
        return to_cuda(torch.tensor([i if i is not None else 0 for i in indexes]))


class Speaker(nn.Module):
    """ Learned continuous representations for binary speaker. Zeros if speaker unknown.
    """

    def __init__(self, speaker_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(3, speaker_dim, padding_idx=0),
            nn.Dropout(0.3)
        )

    def forward(self, speaker_labels):
        """ Embedding table lookup (see src.utils.speaker_label fnc) """
        return self.embeds(to_cuda(torch.tensor(speaker_labels)))
    

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
        if self.distribute_model:
            states = states.cuda(2)
        return states, states

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False
    
class MentionScore(nn.Module):
    """ Mention scoring module
    """
    
    def __init__(self, gi_dim, attn_dim, distance_dim, L, distribute_model):
        super().__init__()

        self.attention = Score(attn_dim)
        self.width = Width(L, distance_dim)
        self.score = Score(gi_dim)
        self.L = L
        self.distribute_model = distribute_model
        
    def forward(self, states, embeds, doc):
        
        #Compute unary mention score for each span
        word2tokens = doc.word2subtok
        
        start_words, end_words, start_toks, end_toks, \
                tok_ranges, word_widths, tok_widths, _ = compute_idx_spans_for_bert(doc.sents, self.L, word2tokens)
        
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
        # Compute span widths (i.e. lengths), embed them
                
        if self.distribute_model:
            widths = self.width(to_cuda(torch.tensor(word_widths), 2))
        else:
            widths = self.width(to_cuda(torch.tensor(word_widths)))
        # Cat it all together to get g_i, our span representation
        g_i = torch.cat((states[start_toks], states[end_toks], attn_embeds, widths), dim=1)
        
        # Compute each span's unary mention score
        mention_scores = self.score(g_i)
        
        # Expriments for coref resolution with all gold mentions
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
        spans = [Span(i1=start_words[idx], 
                      i2=end_words[idx], 
                      id=idx,
                      si=mention_scores[idx],
                      speaker=doc.speaker_start_end(start_words[idx], end_words[idx]), 
                      genre=doc.genre,
                      content=doc.tokens[start_words[idx]:end_words[idx]+1])
                 for idx in indices_sorted]
        
        if self.distribute_model:
            g_i, mention_scores = to_cuda(g_i), to_cuda(mention_scores)
            
        return spans, g_i, mention_scores
                
class HigherOrderScore(nn.Module):
    """ Coreference pair scoring module
    """
    def __init__(self, gij_dim, gi_dim, distance_dim, genre_dim, speaker_dim, K, N):
        super().__init__()
        self.distance = Distance(distance_dim)
        self.distance_coarse = Distance(distance_dim)

        self.genre = Genre(genre_dim)
        self.speaker = Speaker(speaker_dim)
        self.score = Score(gij_dim)
        
        self.coarse_W = nn.Linear(gi_dim, gi_dim, bias=False)
        self.dropout = nn.Dropout(0.3)
                    
        self.W_f = nn.Linear(gi_dim*2, gi_dim)
        self.distance_proj = nn.Linear(distance_dim, 1)
        
        self.bilin = nn.Bilinear(gi_dim, gi_dim, 1)
        
        self.K = K
        self.N = N
        
        
    def forward(self, spans, g_i, mention_scores):
        """ Compute pairwise score for spans and their up to K antecedents
        """
        # ================================================================
        # Second stage: coarse pruning
        # Get the antecedent IDs for current spans
        mention_ids, start_indices, end_indices = zip(*[(span.id, span.i1, span.i2)
                                                        for span in spans])
        
        mention_ids = to_cuda(torch.tensor(mention_ids))
        start_indices = to_cuda(torch.tensor(start_indices))
        end_indices = to_cuda(torch.tensor(end_indices))
        
        i_g = torch.index_select(g_i, 0, mention_ids)
        s_i = torch.index_select(mention_scores, 0, mention_ids)
        
        k = mention_ids.shape[0]
        top_span_range = torch.arange(k)
        antecedent_offsets = to_cuda(top_span_range.unsqueeze(1) - top_span_range.unsqueeze(0))
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
            distances, genres, speakers = zip(*[(i.id, j.id,
                                                i.i2-j.i1, i.genre,
                                                speaker_label(i, j))
                                             for i in spans
                                             for j in i.yi])
        
        # Embed them
        distances = to_cuda(torch.tensor(distances))
        phi = torch.cat((self.distance(distances),
                         self.genre(genres),
                         self.speaker(speakers)), dim=1)
        
        # For indexing a tensor efficiently
        mention_ids = to_cuda(torch.tensor(mention_ids))
        antecedent_ids = to_cuda(torch.tensor(antecedent_ids))
        
        # Get antecedent indexes for each span (first span has no antecedents)
        antecedent_idx = [len(s.yi) for s in spans[1:]]
        unique_mention_ids = to_cuda(torch.tensor([span.id for span in spans[1:]]))     
        epsilon = torch.zeros(unique_mention_ids.shape[0], 1).cuda()
        
        for step in range(self.N):

            # Extract their span representations from the g_i matrix
            i_g = torch.index_select(g_i, 0, mention_ids)
            j_g = torch.index_select(g_i, 0, antecedent_ids)
            
            # Create s_ij(a) representations
            pairs = torch.cat((i_g, j_g, i_g*j_g, phi), dim=1)

            # Score pairs of spans for coreference link
            s_ij_a = self.score(pairs)
            # Compute pairwise scores for coreference links between each mention and
            # its antecedents
            coref_scores = s_ij_a + s_ij_c
            # Split coref scores so each list entry are scores for its antecedents, only.
            # (NOTE that first index is a special case for torch.split, so we handle it here)
            split_scores = [to_cuda(torch.tensor([[0.0]]))] \
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
        scores = torch.cat([torch.zeros(epsilon.shape[0]+1, 1).cuda(), scores], dim=1)
        return spans, scores


class BertCorefScorer(nn.Module):
    """
        Super class to compute coreference links between spans
    """

    def __init__(self, distribute_model=args.distribute_model,
                       attn_dim = args.atten_dim,
                       embeds_dim = args.embeds_dim,
                       distance_dim=args.distance_dim,
                       genre_dim=args.genre_dim,
                       speaker_dim=args.speaker_dim):
        super().__init__()
        assert embeds_dim == 768, 'For bert-based model, embeds_dim should match the size with pre-trained BERT-based Model.'

        self.encoder = BertDocumentEncoder(distribute_model)

        # Forward and backward passes, avg'd attn over embeddings, span width
        gi_dim = attn_dim*2 + embeds_dim + distance_dim
        # gi, gj, gi*gj, distance between gi and gj
        gij_dim = gi_dim*3 + distance_dim + genre_dim + speaker_dim
        
        self.score_spans = MentionScore(gi_dim, attn_dim, distance_dim, L=args.max_span_length, 
                                        distribute_model=distribute_model)
        self.score_pairs = HigherOrderScore(gij_dim, 
                                            gi_dim, distance_dim, genre_dim, 
                                            speaker_dim, K=args.top_K, N=args.higer_order_N)

    def forward(self, doc):
        """
            Encode document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores..
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.encoder(doc)
        
        # Get mention scores for each span, prune
        spans, g_i, mention_scores = self.score_spans(states, embeds, doc)
        # If the document is too short
        if len(spans) <= 2:
            return None, None

        # Get pairwise scores for each span combo
        spans, coref_scores = self.score_pairs(spans, g_i, mention_scores)

        return spans, coref_scores
