from datetime import datetime
import os
import pickle
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np      # type: ignore
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm   # type: ignore
import transformers     # type: ignore

from transformers import BertModel,BertTokenizer
from dataclasses import dataclass
from config import arg_parse
args = arg_parse()

EPSILON = 1e-7
LARGE_VALUE = 1000  # used instead of inf due to bug #16762 in pytorch
Span = Tuple[int, int]

@dataclass
class CorefResult:
    coref_scores: torch.Tensor = None                  # [n_words, k + 1]
    coref_y: torch.Tensor = None                       # [n_words, k + 1]

    word_clusters: List[List[int]] = None
    span_clusters: List[List[Span]] = None

    span_scores: torch.Tensor = None                   # [n_heads, n_words, 2]
    span_y: Tuple[torch.Tensor, torch.Tensor] = None   # [n_heads] x2

class GraphNode:
    def __init__(self, node_id: int):
        self.id = node_id
        self.links: Set[GraphNode] = set()
        self.visited = False

    def link(self, another: "GraphNode"):
        self.links.add(another)
        another.links.add(self)

    def __repr__(self) -> str:
        return str(self.id)
        
class wordLevelCorefScorer(nn.Module):
    """
        Super class to compute coreference links between spans
    """

    def __init__(self):
        super().__init__()
        self.training = True
        if args.bert_name=='bert-base':
            self.bert, _ = BertModel.from_pretrained("bert-base-cased", output_loading_info=True)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        elif args.bert_name=='bert-large':
            self.bert, _ = BertModel.from_pretrained("bert-large-cased", output_loading_info=True)
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
        elif args.bert_name=='spanbert-base':
            self.bert, _ = BertModel.from_pretrained("SpanBERT/spanbert-base-cased", output_loading_info=True)
            self.tokenizer = BertTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
        elif args.bert_name=='spanbert-large':
            self.bert, _ = BertModel.from_pretrained("SpanBERT/spanbert-large-cased", output_loading_info=True)
            self.tokenizer = BertTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")
        elif args.bert_name=='corefbert-base':
            self.bert, _ = BertModel.from_pretrained("nielsr/coref-bert-base", output_loading_info=True)
            self.tokenizer = BertTokenizer.from_pretrained("nielsr/coref-bert-base")
        elif args.bert_name=='corefbert-large':
            self.bert, _ = BertModel.from_pretrained("nielsr/coref-bert-large", output_loading_info=True)
            self.tokenizer = BertTokenizer.from_pretrained("nielsr/coref-bert-large")
        elif args.bert_name=='corefroberta-base':
            self.bert, _ = BertModel.from_pretrained("nielsr/coref-roberta-base", output_loading_info=True)
            self.tokenizer = BertTokenizer.from_pretrained("nielsr/coref-roberta-base")
        elif args.bert_name=='corefroberta-large':
            self.bert, _ = BertModel.from_pretrained("nielsr/coref-roberta-large", output_loading_info=True)
            self.tokenizer = BertTokenizer.from_pretrained("nielsr/coref-roberta-large")
        else:
            raise ValueError('A very specific bad thing happened.')
        
        self.pw = PairwiseEncoder().to(args.device)
        
        bert_emb = args.embeds_dim
        pair_emb = bert_emb * 3 + self.pw.shape

        self.a_scorer = AnaphoricityScorer(pair_emb).to(args.device)
        self.we = WordEncoder(bert_emb).to(args.device)
        self.rough_scorer = RoughScorer(bert_emb).to(args.device)
        self.sp = SpanPredictor(bert_emb, args.distance_dim).to(args.device)

        self.trainable: Dict[str, torch.nn.Module] = {
            "bert": self.bert, "we": self.we,
            "rough_scorer": self.rough_scorer,
            "pw": self.pw, "a_scorer": self.a_scorer,
            "sp": self.sp
        }

    def forward(self,
                doc) -> CorefResult:
        """
        This is a massive method, but it made sense to me to not split it into
        several ones to let one see the data flow.

        Args:
            doc (Doc): a dictionary with the document data.

        Returns:
            CorefResult (see const.py)
        """
        # Encode words with bert
        # words           [n_words, span_emb]
        # cluster_ids     [n_words]
        words, cluster_ids = self.we(doc, self._bertify(doc))

        # Obtain bilinear scores and leave only top-k antecedents for each word
        # top_rough_scores  [n_words, n_ants]
        # top_indices       [n_words, n_ants]
        top_rough_scores, top_indices = self.rough_scorer(words)

        # Get pairwise features [n_words, n_ants, n_pw_features]
        pw = self.pw(top_indices, doc)

        batch_size = args.wl_ascoring_batch_size
        a_scores_lst: List[torch.Tensor] = []

        for i in range(0, len(words), batch_size):
            pw_batch = pw[i:i + batch_size]
            words_batch = words[i:i + batch_size]
            top_indices_batch = top_indices[i:i + batch_size]
            top_rough_scores_batch = top_rough_scores[i:i + batch_size]

            # a_scores_batch    [batch_size, n_ants]
            a_scores_batch = self.a_scorer(
                all_mentions=words, mentions_batch=words_batch,
                pw_batch=pw_batch, top_indices_batch=top_indices_batch,
                top_rough_scores_batch=top_rough_scores_batch
            )
            a_scores_lst.append(a_scores_batch)

        res = CorefResult()

        # coref_scores  [n_spans, n_ants]
        res.coref_scores = torch.cat(a_scores_lst, dim=0)

        res.coref_y = self._get_ground_truth(
            cluster_ids, top_indices, (top_rough_scores > float("-inf")))
        res.word_clusters = self._clusterize(doc, res.coref_scores,
                                             top_indices)
        res.span_scores, res.span_y = self.sp.get_training_data(doc, words)

        # if not self.training:
        res.span_clusters = self.sp.predict(doc, words, res.word_clusters)

        return res

    def _bertify(self, doc) -> torch.Tensor:
        subwords_batches = get_subwords_batches(doc, self.tokenizer)

        special_tokens = np.array([self.tokenizer.cls_token_id,
                                   self.tokenizer.sep_token_id,
                                   self.tokenizer.pad_token_id])
        subword_mask = ~(np.isin(subwords_batches, special_tokens))

        subwords_batches_tensor = torch.tensor(subwords_batches,
                                               device=args.device,
                                               dtype=torch.long)
        subword_mask_tensor = torch.tensor(subword_mask,
                                           device=args.device)

        # Obtain bert output for selected batches only
        attention_mask = (subwords_batches != self.tokenizer.pad_token_id)
        out = self.bert(subwords_batches_tensor, attention_mask=torch.tensor(attention_mask, device=args.device))[0]

        # [n_subwords, bert_emb]
        return out[subword_mask_tensor]
    
    @staticmethod
    def _get_ground_truth(cluster_ids: torch.Tensor,
                          top_indices: torch.Tensor,
                          valid_pair_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cluster_ids: tensor of shape [n_words], containing cluster indices
                for each word. Non-gold words have cluster id of zero.
            top_indices: tensor of shape [n_words, n_ants],
                indices of antecedents of each word
            valid_pair_map: boolean tensor of shape [n_words, n_ants],
                whether for pair at [i, j] (i-th word and j-th word)
                j < i is True

        Returns:
            tensor of shape [n_words, n_ants + 1] (dummy added),
                containing 1 at position [i, j] if i-th and j-th words corefer.
        """
        y = cluster_ids[top_indices] * valid_pair_map  # [n_words, n_ants]
        y[y == 0] = -1                                 # -1 for non-gold words
        y = add_dummy(y)                         # [n_words, n_cands + 1]
        y = (y == cluster_ids.unsqueeze(1))            # True if coreferent
        # For all rows with no gold antecedents setting dummy to True
        y[y.sum(dim=1) == 0, 0] = True
        return y.to(torch.float)
    
    def _clusterize(self, doc, scores: torch.Tensor, top_indices: torch.Tensor):
        antecedents = scores.argmax(dim=1) - 1
        not_dummy = antecedents >= 0
        coref_span_heads = torch.arange(0, len(scores))[not_dummy]
        antecedents = top_indices[coref_span_heads, antecedents[not_dummy]]

        nodes = [GraphNode(i) for i in range(len(doc.cased_words))]
        for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
            nodes[i].link(nodes[j])
            assert nodes[i] is not nodes[j]

        clusters = []
        for node in nodes:
            if len(node.links) > 0 and not node.visited:
                cluster = []
                stack = [node]
                while stack:
                    current_node = stack.pop()
                    current_node.visited = True
                    cluster.append(current_node.id)
                    stack.extend(link for link in current_node.links if not link.visited)
                assert len(cluster) > 1
                clusters.append(sorted(cluster))
        return sorted(clusters)


class PairwiseEncoder(torch.nn.Module):
    """ A Pytorch module to obtain feature embeddings for pairwise features

    Usage:
        encoder = PairwiseEncoder(config)
        pairwise_features = encoder(pair_indices, doc)
    """
    def __init__(self):
        super().__init__()
        emb_size = args.genre_dim

        self.genre2int = {g: gi for gi, g in enumerate(["sec", "bn", "mz", "nw",
                                                        "pt", "tc", "wb"])}
        self.genre_emb = torch.nn.Embedding(len(self.genre2int), emb_size)

        # each position corresponds to a bucket:
        #   [(0, 2), (2, 3), (3, 4), (4, 5), (5, 8),
        #    (8, 16), (16, 32), (32, 64), (64, float("inf"))]
        self.distance_emb = torch.nn.Embedding(9, emb_size)

        # two possibilities: same vs different speaker
        self.speaker_emb = torch.nn.Embedding(2, emb_size)

        self.dropout = torch.nn.Dropout(args.dropout_rate)
        self.shape = emb_size * 3  # genre, distance, speaker\

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.genre_emb.parameters()).device

    def forward(self,
                top_indices,
                doc) -> torch.Tensor:
        word_ids = torch.arange(0, len(doc.cased_words), device=self.device)
        speaker_map = torch.tensor(self._speaker_map(doc), device=self.device)

        same_speaker = (speaker_map[top_indices] == speaker_map.unsqueeze(1))
        same_speaker = self.speaker_emb(same_speaker.to(torch.long))

        # bucketing the distance (see __init__())
        distance = (word_ids.unsqueeze(1) - word_ids[top_indices]
                    ).clamp_min_(min=1)
        log_distance = distance.to(torch.float).log2().floor_()
        log_distance = log_distance.clamp_max_(max=6).to(torch.long)
        distance = torch.where(distance < 5, distance - 1, log_distance + 2)
        distance = self.distance_emb(distance)

        genre = torch.tensor(self.genre2int['sec'],
                             device=self.device).expand_as(top_indices)
        genre = self.genre_emb(genre)

        return self.dropout(torch.cat((same_speaker, distance, genre), dim=2))

    @staticmethod
    def _speaker_map(doc) -> List[int]:
        """
        Returns a tensor where i-th element is the speaker id of i-th word.
        """
        # speaker string -> speaker id
        str2int = {s: i for i, s in enumerate(set(doc.speakers))}

        # word id -> speaker id
        return [str2int[s] for s in doc.speakers]


class RoughScorer(torch.nn.Module):
    """
    Is needed to give a roughly estimate of the anaphoricity of two candidates,
    only top scoring candidates are considered on later steps to reduce
    computational complexity.
    """
    def __init__(self, features: int):
        super().__init__()
        self.dropout = torch.nn.Dropout(args.dropout_rate)
        self.bilinear = torch.nn.Linear(features, features)

        self.k = args.top_K

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                mentions: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns rough anaphoricity scores for candidates, which consist of
        the bilinear output of the current model summed with mention scores.
        """
        # [n_mentions, n_mentions]
        pair_mask = torch.arange(mentions.shape[0])
        pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        pair_mask = torch.log((pair_mask > 0).to(torch.float))
        pair_mask = pair_mask.to(mentions.device)

        bilinear_scores = self.dropout(self.bilinear(mentions)).mm(mentions.T)

        rough_scores = pair_mask + bilinear_scores

        return self._prune(rough_scores)

    def _prune(self,
               rough_scores: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects top-k rough antecedent scores for each mention.

        Args:
            rough_scores: tensor of shape [n_mentions, n_mentions], containing
                rough antecedent scores of each mention-antecedent pair.

        Returns:
            FloatTensor of shape [n_mentions, k], top rough scores
            LongTensor of shape [n_mentions, k], top indices
        """
        top_scores, indices = torch.topk(rough_scores,
                                         k=min(self.k, len(rough_scores)),
                                         dim=1, sorted=False)
        return top_scores, indices


class AnaphoricityScorer(torch.nn.Module):
    """ Calculates anaphoricity scores by passing the inputs into a FFNN """
    def __init__(self,
                 in_features: int):
        super().__init__()
        hidden_size = args.wl_anaphoricity_hidden_size
        if not args.wl_n_hidden_layer:
            hidden_size = in_features
        layers = []
        for i in range(args.wl_n_hidden_layer):
            layers.extend([torch.nn.Linear(hidden_size if i else in_features,
                                           hidden_size),
                           torch.nn.LeakyReLU(),
                           torch.nn.Dropout(args.dropout_rate)])
        self.hidden = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(hidden_size, out_features=1)

    def forward(self, *,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                all_mentions: torch.Tensor,
                mentions_batch: torch.Tensor,
                pw_batch: torch.Tensor,
                top_indices_batch: torch.Tensor,
                top_rough_scores_batch: torch.Tensor,
                ) -> torch.Tensor:
        """ Builds a pairwise matrix, scores the pairs and returns the scores.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb]
            mentions_batch (torch.Tensor): [batch_size, mention_emb]
            pw_batch (torch.Tensor): [batch_size, n_ants, pw_emb]
            top_indices_batch (torch.Tensor): [batch_size, n_ants]
            top_rough_scores_batch (torch.Tensor): [batch_size, n_ants]

        Returns:
            torch.Tensor [batch_size, n_ants + 1]
                anaphoricity scores for the pairs + a dummy column
        """
        # [batch_size, n_ants, pair_emb]
        pair_matrix = self._get_pair_matrix(
            all_mentions, mentions_batch, pw_batch, top_indices_batch)

        # [batch_size, n_ants]
        scores = top_rough_scores_batch + self._ffnn(pair_matrix)
        scores = add_dummy(scores, eps=True)

        return scores

    def _ffnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates anaphoricity scores.

        Args:
            x: tensor of shape [batch_size, n_ants, n_features]

        Returns:
            tensor of shape [batch_size, n_ants]
        """
        x = self.out(self.hidden(x))
        return x.squeeze(2)

    @staticmethod
    def _get_pair_matrix(all_mentions: torch.Tensor,
                         mentions_batch: torch.Tensor,
                         pw_batch: torch.Tensor,
                         top_indices_batch: torch.Tensor,
                         ) -> torch.Tensor:
        """
        Builds the matrix used as input for AnaphoricityScorer.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb],
                all the valid mentions of the document,
                can be on a different device
            mentions_batch (torch.Tensor): [batch_size, mention_emb],
                the mentions of the current batch,
                is expected to be on the current device
            pw_batch (torch.Tensor): [batch_size, n_ants, pw_emb],
                pairwise features of the current batch,
                is expected to be on the current device
            top_indices_batch (torch.Tensor): [batch_size, n_ants],
                indices of antecedents of each mention

        Returns:
            torch.Tensor: [batch_size, n_ants, pair_emb]
        """
        emb_size = mentions_batch.shape[1]
        n_ants = pw_batch.shape[1]

        a_mentions = mentions_batch.unsqueeze(1).expand(-1, n_ants, emb_size)
        b_mentions = all_mentions[top_indices_batch]
        similarity = a_mentions * b_mentions

        out = torch.cat((a_mentions, b_mentions, similarity, pw_batch), dim=2)
        return out


class WordEncoder(torch.nn.Module):
    """ Receives bert contextual embeddings of a text, extracts all the
    possible mentions in that text. """

    def __init__(self, features: int):
        """
        Args:
            features (int): the number of featues in the input embeddings
            config (Config): the configuration of the current session
        """
        super().__init__()
        self.attn = torch.nn.Linear(in_features=features, out_features=1)
        self.dropout = torch.nn.Dropout(args.dropout_rate)

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.attn.parameters()).device

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                doc,
                x: torch.Tensor,
                ) -> Tuple[torch.Tensor, ...]:
        """
        Extracts word representations from text.

        Args:
            doc: the document data
            x: a tensor containing bert output, shape (n_subtokens, bert_dim)

        Returns:
            words: a Tensor of shape [n_words, mention_emb];
                mention representations
            cluster_ids: tensor of shape [n_words], containing cluster indices
                for each word. Non-coreferent words have cluster id of zero.
        """
        word_boundaries = torch.tensor(doc.word2subword, device=self.device)
        starts = word_boundaries[:, 0]
        ends = word_boundaries[:, 1]

        # [n_mentions, features]
        words = self._attn_scores(x, starts, ends).mm(x)

        words = self.dropout(words)

        return (words, self._cluster_ids(doc))

    def _attn_scores(self,
                     bert_out: torch.Tensor,
                     word_starts: torch.Tensor,
                     word_ends: torch.Tensor) -> torch.Tensor:
        """ Calculates attention scores for each of the mentions.

        Args:
            bert_out (torch.Tensor): [n_subwords, bert_emb], bert embeddings
                for each of the subwords in the document
            word_starts (torch.Tensor): [n_words], start indices of words
            word_ends (torch.Tensor): [n_words], end indices of words

        Returns:
            torch.Tensor: [description]
        """
        n_subtokens = len(bert_out)
        n_words = len(word_starts)

        # [n_mentions, n_subtokens]
        # with 0 at positions belonging to the words and -inf elsewhere
        attn_mask = torch.arange(0, n_subtokens, device=self.device).expand((n_words, n_subtokens))
        attn_mask = ((attn_mask >= word_starts.unsqueeze(1))
                     * (attn_mask < word_ends.unsqueeze(1)))
        attn_mask = torch.log(attn_mask.to(torch.float))

        attn_scores = self.attn(bert_out).T  # [1, n_subtokens]
        attn_scores = attn_scores.expand((n_words, n_subtokens))
        attn_scores = attn_mask + attn_scores
        del attn_mask
        return torch.softmax(attn_scores, dim=1)  # [n_words, n_subtokens]

    def _cluster_ids(self, doc) -> torch.Tensor:
        """
        Args:
            doc: document information

        Returns:
            torch.Tensor of shape [n_word], containing cluster indices for
                each word. Non-coreferent words have cluster id of zero.
        """
        word2cluster = {word_i: i
                        for i, cluster in enumerate(doc.word_clusters, start=1)
                        for word_i in cluster}

        return torch.tensor(
            [word2cluster.get(word_i, 0)
             for word_i in range(len(doc.cased_words))],
            device=self.device
        )


class SpanPredictor(torch.nn.Module):
    def __init__(self, input_size: int, distance_emb_size: int):
        super().__init__()
        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(input_size * 2 + 64, input_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
        )
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(64, 4, 3, 1, 1),
            torch.nn.Conv1d(4, 2, 3, 1, 1)
        )
        self.emb = torch.nn.Embedding(128, distance_emb_size) # [-63, 63] + too_far

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.ffnn.parameters()).device

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                doc,
                words: torch.Tensor,
                heads_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculates span start/end scores of words for each span head in
        heads_ids

        Args:
            doc (Doc): the document data
            words (torch.Tensor): contextual embeddings for each word in the
                document, [n_words, emb_size]
            heads_ids (torch.Tensor): word indices of span heads

        Returns:
            torch.Tensor: span start/end scores, [n_heads, n_words, 2]
        """
        # Obtain distance embedding indices, [n_heads, n_words]
        relative_positions = (heads_ids.unsqueeze(1) - torch.arange(words.shape[0], device=words.device).unsqueeze(0))
        emb_ids = relative_positions + 63               # make all valid distances positive
        emb_ids[(emb_ids < 0) + (emb_ids > 126)] = 127  # "too_far"

        # Obtain "same sentence" boolean mask, [n_heads, n_words]
        sent_id = torch.tensor(doc.sent_id, device=words.device)
        same_sent = (sent_id[heads_ids].unsqueeze(1) == sent_id.unsqueeze(0))

        # To save memory, only pass candidates from one sentence for each head
        # pair_matrix contains concatenated span_head_emb + candidate_emb + distance_emb
        # for each candidate among the words in the same sentence as span_head
        # [n_heads, input_size * 2 + distance_emb_size]
        rows, cols = same_sent.nonzero(as_tuple=True)
        pair_matrix = torch.cat((
            words[heads_ids[rows]],
            words[cols],
            self.emb(emb_ids[rows, cols]),
        ), dim=1)

        lengths = same_sent.sum(dim=1)
        padding_mask = torch.arange(0, lengths.max(), device=words.device).unsqueeze(0)
        padding_mask = (padding_mask < lengths.unsqueeze(1))  # [n_heads, max_sent_len]

        # [n_heads, max_sent_len, input_size * 2 + distance_emb_size]
        # This is necessary to allow the convolution layer to look at several
        # word scores
        padded_pairs = torch.zeros(*padding_mask.shape, pair_matrix.shape[-1], device=words.device)
        padded_pairs[padding_mask] = pair_matrix

        res = self.ffnn(padded_pairs) # [n_heads, n_candidates, last_layer_output]
        res = self.conv(res.permute(0, 2, 1)).permute(0, 2, 1) # [n_heads, n_candidates, 2]

        scores = torch.full((heads_ids.shape[0], words.shape[0], 2), float('-inf'), device=words.device)
        scores[rows, cols] = res[padding_mask]

        ## apply softmax
        scores = F.softmax(scores, dim=1)

        # Make sure that start <= head <= end during inference
        if not self.training:
            valid_starts = torch.log((relative_positions >= 0).to(torch.float))
            valid_ends = torch.log((relative_positions <= 0).to(torch.float))
            valid_positions = torch.stack((valid_starts, valid_ends), dim=2)
            return scores + valid_positions
        
        return scores

    def get_training_data(self,
                          doc,
                          words: torch.Tensor
                          ) -> Tuple[Optional[torch.Tensor],
                                     Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """ Returns span starts/ends for gold mentions in the document. """
        head2span = sorted(doc.head2span)
        if not head2span:
            return None, None
        heads, starts, ends = zip(*head2span)
        heads = torch.tensor(heads, device=self.device)
        starts = torch.tensor(starts, device=self.device)
        ends = torch.tensor(ends, device=self.device) - 1
        return self(doc, words, heads), (starts, ends)

    def predict(self,
                doc,
                words: torch.Tensor,
                clusters):
        """
        Predicts span clusters based on the word clusters.

        Args:
            doc (Doc): the document data
            words (torch.Tensor): [n_words, emb_size] matrix containing
                embeddings for each of the words in the text
            clusters (List[List[int]]): a list of clusters where each cluster
                is a list of word indices

        Returns:
            List[List[Span]]: span clusters
        """
        if not clusters:
            return []

        heads_ids = torch.tensor(
            sorted(i for cluster in clusters for i in cluster),
            device=self.device
        )

        scores = self(doc, words, heads_ids)
        starts = scores[:, :, 0].argmax(dim=1).tolist()
        ends = (scores[:, :, 1].argmax(dim=1) + 1).tolist()

        head2span = {
            head: (start, end)
            for head, start, end in zip(heads_ids.tolist(), starts, ends)
        }

        return [[head2span[head] for head in cluster]
                for cluster in clusters]




def add_dummy(tensor: torch.Tensor, eps: bool = False):
    """ Prepends zeros (or a very small value if eps is True)
    to the first (not zeroth) dimension of tensor.
    """
    kwargs = dict(device=tensor.device, dtype=tensor.dtype)
    shape: List[int] = list(tensor.shape)
    shape[1] = 1
    if not eps:
        dummy = torch.zeros(shape, **kwargs)          # type: ignore
    else:
        dummy = torch.full(shape, EPSILON, **kwargs)  # type: ignore
    return torch.cat((dummy, tensor), dim=1)

def get_subwords_batches(doc,
                         tok) -> np.ndarray:
    """
    Turns a list of subwords to a list of lists of subword indices
    of max length == batch_size (or shorter, as batch boundaries
    should match sentence boundaries). Each batch is enclosed in cls and sep
    special tokens.

    Returns:
        batches of bert tokens [n_batches, batch_size]
    """
    batch_size = args.wl_bert_window_size - 2  # to save space for CLS and SEP

    subwords: List[str] = doc.subwords
    subwords_batches = []
    start, end = 0, 0

    while end < len(subwords):
        end = min(end + batch_size, len(subwords))

        # Move back till we hit a sentence end
        if end < len(subwords):
            sent_id = doc.sent_id[doc.word_id[end]]
            while end and doc.sent_id[doc.word_id[end - 1]] == sent_id:
                end -= 1

        length = end - start
        batch = [tok.cls_token] + subwords[start:end] + [tok.sep_token]
        batch_ids = [-1] + list(range(start, end)) + [-1]

        # Padding to desired length
        # -1 means the token is a special token
        batch += [tok.pad_token] * (batch_size - length)
        batch_ids += [-1] * (batch_size - length)

        subwords_batches.append([tok.convert_tokens_to_ids(token)
                                 for token in batch])
        start += length

    return np.array(subwords_batches)