# from msilib.schema import Error
from fileinput import filename
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizer

import os, io, re, attr, random
from collections import defaultdict
from fnmatch import fnmatch
from copy import deepcopy as c
import pickle
import re
import random
from tqdm import tqdm

from utils import *
import nltk

from nltk.tokenize import sent_tokenize
from nltk.tokenize import WordPunctTokenizer

from collections import Counter
from config import arg_parse

import stanza

args = arg_parse()

SPEAKER_START = '[unused19]'
SPEAKER_END = '[unused73]'

# Load bert tokenizer once
if args.bert_based:
    global bert_tokenizer

    if args.bert_name=='bert-base':
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    elif args.bert_name=='bert-large':
        bert_tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    elif args.bert_name=='spanbert-base':
        # bert_tokenizer = BertTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
        bert_tokenizer = BertTokenizer.from_pretrained("BERTs/spanbert-base-cased")
    elif args.bert_name=='spanbert-large':
        bert_tokenizer = BertTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")
    elif args.bert_name=='corefbert-base':
        bert_tokenizer = BertTokenizer.from_pretrained("nielsr/coref-bert-base")
    elif args.bert_name=='corefbert-large':
        bert_tokenizer = BertTokenizer.from_pretrained("nielsr/coref-bert-large")
    elif args.bert_name=='corefroberta-base':
        bert_tokenizer = BertTokenizer.from_pretrained("nielsr/coref-roberta-base")
    elif args.bert_name=='corefroberta-large':
        bert_tokenizer = BertTokenizer.from_pretrained("nielsr/coref-roberta-large")
    else:
        raise ValueError('Cannot find the right bert version.')
        
if args.model in ['wordLevelModel', 'cyberCorefModel']:
        global syntax_parser
        # https://github.com/stanfordnlp/stanza/issues/275
        # syntax_parser = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True, dir='/home/featurize/stanza_resources')
        syntax_parser = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True,download_method=2, dir='/home/featurize/work/stanza_resources')

class Corpus:
    def __init__(self, documents):
        self.docs = documents
        self.vocab, self.char_vocab = self.get_vocab()

    def __getitem__(self, idx):
        return self.docs[idx]

    def __len__(self):
        return len(self.docs) 

    def __repr__(self):
        return 'Corpus containg %d documents' % len(self.docs)

    def get_vocab(self):
        """ Set vocabulary for LazyVectors """
        vocab, char_vocab = set(), set()

        for document in self.docs:
            vocab.update(document.tokens)
            char_vocab.update([char
                               for word in document.tokens
                               for char in word])

        return vocab, char_vocab

class Document:
    def __init__(self, raw_text, tokens, sents, corefs, speakers, genre, filename, entities):
        self.raw_text = raw_text
        self.tokens = tokens
        self.sents = sents
        self.corefs = corefs
        self.speakers = speakers
        self.genre = genre
        self.filename = filename
        self.entities = entities

        # Filled in at evaluation time.
        self.tags = None

    def __getitem__(self, idx):
        return (self.tokens[idx], self.corefs[idx], \
                self.speakers[idx], self.genre)

    def __repr__(self):
        return 'Document containing %d tokens' % len(self.tokens)

    def __len__(self):
        return len(self.tokens)
    
    def spans(self):
            """ Create Span object for each span """
            return [Span(i1=i[0], i2=i[-1], id=idx,
                        speaker=self.speaker(i), genre=self.genre)
                    for idx, i in enumerate(compute_idx_spans(self.sents))]

    def truncate(self):
        """ 
        自裁
        Randomly truncate the document to up to MAX sentences 
        """
        MAX = args.sentense_max_num
        if len(self.sents) > MAX:
            # num_sents >= i >= MAX
            i = random.sample(range(MAX, len(self.sents)), 1)[0]
            tokens = flatten(self.sents[i-MAX:i])

            pre_sents = self.sents[0:i-MAX]
            pre_tokens = flatten(pre_sents)
            # Index of first token in truncated sentences
            num_pre_tokens = len(pre_tokens)
            # Index of last token in truncated sentences
            num_pre_curr_tokens = num_pre_tokens + len(tokens) - 1
            new_corefs = []

            for coref in self.corefs:
                # Ignore corefs outside of current sentences
                if coref['start'] < num_pre_tokens or coref['end'] > num_pre_curr_tokens:
                    continue
                new_start = coref['start'] - num_pre_tokens
                new_end = coref['end'] - num_pre_tokens
                new_coref = {'label': coref['label'], 
                                'start': new_start,
                                'end': new_end,
                                'span': (new_start, new_end)}
                new_corefs.append(new_coref)

            new_speakers = self.speakers[num_pre_tokens:num_pre_curr_tokens + 1]

            return self.__class__(c(self.raw_text), tokens, self.sents[i-MAX:i],
                                    new_corefs, new_speakers,
                                    c(self.genre), c(self.filename))
        return self

    def speaker(self, i):
        """ Compute speaker of a span """
        if self.speakers[i[0]] == self.speakers[i[-1]]:
            return self.speakers[i[0]]
        return None
    
    def speaker_start_end(self, start, end):
        """ Compute speaker of a span """
        if self.speakers[start] == self.speakers[end]:
            return self.speakers[start]
        return None

class BERTDocument(Document):
    def __init__(self, raw_text, tokens, sents, corefs, speakers, genre, filename, entities):
        super().__init__(raw_text, tokens, sents, corefs, speakers, genre, filename, entities)

        self.bert_tokens = []
        self.word_subtoken_map = [] # nltk_token2bert_token
        self.word2idx = {}
        self.segment_subtoken_map = []
        self.segments = []
        
        self.sentence_ends, self.token_ends, self.sentence_ends_subtok = [], [], []
        self.word2sent = []
        self.sent2subtok_bdry = []

        if args.bert_based and args.model != 'corefQAModel':
            self._compute_bert_tokenization()
        else:
            raise ValueError('You should choose a different Document class or you should revise the configs.')
        
    def _compute_bert_tokenization(self):
        """
        生成bert_sub_tokens
        
        self.bert_tokens: List, 将nltk分词后生成token重新进行tokenize, 转换为bert的token字典中的索引
        self.word_subtoken_map: List, size:len(bert_tokens), 每个bert_token对应的word
        self.word2subtok: Dict, key:token_idx values:sub_token_idx, token到sub_token的转换
        self.segments: List, size:segment_num, item:Tensor, 将文章分为若干小于max_segment_len的segment
        self.segment_subtoken_map: List, size:len(subtokens)+2*segment_num, 在word_subtoken_map基础上加入[CLS][SEP]位置
        """
        global bert_tokenizer
        
        max_segment_len = int(args.max_segment_len)
        
        idx_word = 0
        idx_subtoken = 0
        
        for j, sent in enumerate(self.sents):
            for i, token in enumerate(sent):
                
                subtokens = bert_tokenizer(token)['input_ids'][1:-1]
                
                self.bert_tokens.extend(subtokens)
                
                self.token_ends.extend([False] * len(subtokens))
                self.token_ends[-1] = True

                self.sentence_ends.extend([False] * len(subtokens))
                
                self.word_subtoken_map.extend([idx_word] * len(subtokens))
                self.word2sent.append(j)
                idx_word += 1
                idx_subtoken += len(subtokens)
                
            self.sentence_ends[-1] = True
        
        # 将doc按照max_segment_len进行分割，保证分割部分为句的结束
        current = 0
        previous_token = 0
        
        while current < len(self.bert_tokens):
            # Min of last token of the document, or 
            # - 2 to include [CLS] and [SEP] tokens, -1 to refer to the corrent arr element
            end = min(current + max_segment_len - 1 - 2, len(self.bert_tokens) - 1)

            while end >= current and not self.sentence_ends[end]:
                # make end of segment be end of some sentence, or equal to current token
                end -= 1
            # How can end be less than current? Only if it is less by 1 (previous constraint not satisfied)
            if end < current:
                # Put the end token back?
                end = min(current + max_segment_len - 1 - 2,  len(self.bert_tokens) - 1)
                assert self.word2sent[self.word_subtoken_map[current]] == self.word2sent[self.word_subtoken_map[end]]
                # Make the end be end of last token
                while end >= current and not self.token_ends[end]:
                    end -= 1
                if end < current:
                    raise Exception("Can't find valid segment")
                    
            # Make segment consist of subtokens for found boundaries
            self.segments.append(torch.LongTensor([101] + self.bert_tokens[current : end + 1] + [102]))
            
            subtoken_map = self.word_subtoken_map[current : end + 1]
            
            # Make the [CLS] token of the segment map to last word of previous segment and [SEP] token
            # to last word in the current segment.
            self.segment_subtoken_map.extend([previous_token] + subtoken_map + [subtoken_map[-1]])
            
            subtoken_sent_ends = self.sentence_ends[current : end + 1]
            subtoken_sent_ends[-1] = False
            self.sentence_ends_subtok.extend([False] + subtoken_sent_ends + [True])
                
            current = end + 1
            previous_token = subtoken_map[-1]
            
            
        self.word2subtok = defaultdict(list)
        sentence_idx = 0
        for i, word_idx in enumerate(self.segment_subtoken_map):
            self.word2subtok[word_idx].append(i)
            # If current token is an end of sentence
            if self.sentence_ends_subtok[i]:
                self.sent2subtok_bdry.append((sentence_idx, i))
                sentence_idx = i+1


    def truncate(self):
        """ 
        子类重构自裁
        Randomly truncate the document to up to MAX sentences 
        """
        MAX = args.segment_max_num
        if len(self.segments) > MAX: 
            # 以i作为终止，随机选择MAX个segments进行训练
            i = random.sample(range(MAX, len(self.segments)), 1)[0]
            subtokens = flatten(self.segments[i-MAX:i])
            
            
            # Index of the first token in the truncated segments
            num_pre_subtokens = len(flatten(self.segments[0:i-MAX]))
            # Index of the last token in the truncated segments
            num_pre_curr_subtokens = num_pre_subtokens + len(subtokens) -1
            
            # Index of the first and the last word corresponding to 
            # given truncated segments
            first_word_idx, last_word_idx = self.segment_subtoken_map[num_pre_subtokens], \
                                                self.segment_subtoken_map[num_pre_curr_subtokens]
            
            first_sentence_idx, last_sentence_idx = self.word2sent[first_word_idx], \
                                                        self.word2sent[last_word_idx]
            sents = self.sents[first_sentence_idx:last_sentence_idx + 1]
                # +1 to include last sentence too
            tokens = flatten(sents)
            pre_sents = self.sents[0:first_sentence_idx]
            pre_tokens = flatten(pre_sents)
            # Index of first token in truncated sentences
            num_pre_tokens = len(pre_tokens)
            # Index of last token in truncated sentences
            num_pre_curr_tokens = num_pre_tokens + len(tokens) - 1
            new_corefs = []

            for coref in self.corefs:
                # Ignore corefs outside of current sentences
                if coref['start'] < num_pre_tokens or coref['end'] > num_pre_curr_tokens:
                    continue
                new_start = coref['start'] - num_pre_tokens
                new_end = coref['end'] - num_pre_tokens
                new_coref = {'label': coref['label'], 
                                'start': new_start,
                                'end': new_end,
                                'span': (new_start, new_end)}
                new_corefs.append(new_coref)

            new_speakers = self.speakers[num_pre_tokens:num_pre_curr_tokens + 1]

            return self.__class__(c(self.raw_text), tokens, sents,
                                    new_corefs, new_speakers,
                                    c(self.genre), c(self.filename), c(self.entities))
        return self

class corefQADocument(Document):
    def __init__(self, raw_text, tokens, sents, corefs, speakers, genre, filename, entities):
        """
        计算corefQA模型所需要的变量
        参考https://github.com/YuxianMeng/CorefQA-pytorch/dataloder/conll_data_processor.py
        """
        super().__init__(raw_text, tokens, sents, corefs, speakers, genre, filename, entities)

        coref_cluster_num = max([coref['label'] for coref in self.corefs])+1
        doc_info = {
            'doc_key': self.filename,
            'sentences': self.sents,
            'speakers': [],
            'clusters': [[] for _ in range(coref_cluster_num)]
        }
       
        for coref in self.corefs:
            doc_info['clusters'][coref['label']].append(coref['span'])

        tokenized_document = self._corefQA_tokenize_document(doc_info, bert_tokenizer, max_doc_length=1000)
        token_windows, mask_windows = self._convert_to_sliding_window_corefQA(tokenized_document, sliding_window_size=128)
        input_id_windows = [bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in token_windows]

        span_start, span_end, mention_span, cluster_ids = self._flatten_clusters_corefQA(tokenized_document['clusters'])

        self.doc_idx = self.filename
        self.sentence_map = tokenized_document['sentence_map']
        self.subtoken_map = tokenized_document['subtoken_map']
        self.flattened_window_input_ids = input_id_windows
        self.flattened_window_masked_ids = mask_windows
        self.span_start = span_start
        self.span_end = span_end
        self.mention_span = mention_span
        self.cluster_ids = cluster_ids

    def _corefQA_tokenize_document(self, doc_info, tokenizer=BertTokenizer, max_doc_length=1000):
        """
        tokenize into sub tokens
        :param doc_info:
        :param tokenizer:
        max_doc_length: pad to max_doc_length
        :return:
        """
        sub_tokens = []  # all sub tokens of a document
        sentence_map = []  # collected tokenized tokens -> sentence id
        subtoken_map = []  # collected tokenized tokens -> original token id

        word_idx = -1

        for sentence_id, sentence in enumerate(doc_info['sentences']):
            for token in sentence:
                word_idx += 1
                word_tokens = tokenizer.tokenize(token)
                sub_tokens.extend(word_tokens)
                sentence_map.extend([sentence_id] * len(word_tokens))
                subtoken_map.extend([word_idx] * len(word_tokens))
        if max_doc_length:
            num_to_pad = max_doc_length - len(sub_tokens)
            sub_tokens.extend(["[PAD]"] * num_to_pad)
            sentence_map.extend([sentence_map[-1]+1] * num_to_pad)
            subtoken_map.extend(list(range(word_idx+1, num_to_pad+1+word_idx)))
 
        speakers = {subtoken_map.index(word_index): tokenizer.tokenize(speaker)
                    for word_index, speaker in doc_info['speakers']}
        clusters = [[(subtoken_map.index(start), len(subtoken_map) - 1 - subtoken_map[::-1].index(end))
                    for start, end in cluster] for cluster in doc_info['clusters']]
        tokenized_document = {'sub_tokens': sub_tokens, 'sentence_map': sentence_map, 'subtoken_map': subtoken_map,
                            'speakers': speakers, 'clusters': clusters, 'doc_key': doc_info['doc_key']}
        return tokenized_document

    def _convert_to_sliding_window_corefQA(self, tokenized_document, sliding_window_size):
        """
        construct sliding windows, allocate tokens and masks into each window
        :param tokenized_document:
        :param sliding_window_size:
        :return:
        """
        expanded_tokens, expanded_masks = self._expand_with_speakers_corefQA(tokenized_document)
        sliding_windows = self._construct_sliding_windows_corefQA(len(expanded_tokens), sliding_window_size - 2)
        token_windows = []  # expanded tokens to sliding window
        mask_windows = []  # expanded masks to sliding window
        for window_start, window_end, window_mask in sliding_windows:
            original_tokens = expanded_tokens[window_start: window_end]
            original_masks = expanded_masks[window_start: window_end]
            window_masks = [-2 if w == 0 else o for w, o in zip(window_mask, original_masks)]
            one_window_token = ['[CLS]'] + original_tokens + ['[SEP]'] + ['[PAD]'] * (
                    sliding_window_size - 2 - len(original_tokens))
            one_window_mask = [-3] + window_masks + [-3] + [-4] * (sliding_window_size - 2 - len(original_tokens))
            assert len(one_window_token) == sliding_window_size
            assert len(one_window_mask) == sliding_window_size
            token_windows.append(one_window_token)
            mask_windows.append(one_window_mask)
        assert len(tokenized_document['sentence_map']) == sum([i >= 0 for j in mask_windows for i in j])
        return token_windows, mask_windows
    
    def _expand_with_speakers_corefQA(self, tokenized_document):
        """
        add speaker name information
        :param tokenized_document: tokenized document information
        :return:
        """
        expanded_tokens = []
        expanded_masks = []
        for token_idx, token in enumerate(tokenized_document['sub_tokens']):
            if token_idx in tokenized_document['speakers']:
                speaker = [SPEAKER_START] + tokenized_document['speakers'][token_idx] + [SPEAKER_END]
                expanded_tokens.extend(speaker)
                expanded_masks.extend([-1] * len(speaker))
            expanded_tokens.append(token)
            expanded_masks.append(token_idx)
        return expanded_tokens, expanded_masks
    
    def _construct_sliding_windows_corefQA(self, sequence_length, sliding_window_size):
        """
        construct sliding windows for BERT processing
        :param sequence_length: e.g. 9
        :param sliding_window_size: e.g. 4
        :return: [(0, 4, [1, 1, 1, 0]), (2, 6, [0, 1, 1, 0]), (4, 8, [0, 1, 1, 0]), (6, 9, [0, 1, 1])]
        """
        sliding_windows = []
        stride = int(sliding_window_size / 2)
        start_index = 0
        end_index = 0
        while end_index < sequence_length:
            end_index = min(start_index + sliding_window_size, sequence_length)
            left_value = 1 if start_index == 0 else 0
            right_value = 1 if end_index == sequence_length else 0
            mask = [left_value] * int(sliding_window_size / 4) + [1] * int(sliding_window_size / 2) \
                + [right_value] * (sliding_window_size - int(sliding_window_size / 2) - int(sliding_window_size / 4))
            mask = mask[: end_index - start_index]
            sliding_windows.append((start_index, end_index, mask))
            start_index += stride
        assert sum([sum(window[2]) for window in sliding_windows]) == sequence_length
        return sliding_windows
    
    def _flatten_clusters_corefQA(self, clusters):
        """
        flattern cluster information
        :param clusters:
        :return:
        """
        span_starts = []
        span_ends = []
        cluster_ids = []
        mention_span = []
        for cluster_id, cluster in enumerate(clusters):
            for start, end in cluster:
                span_starts.append(start)
                span_ends.append(end)
                mention_span.append((start, end))
                cluster_ids.append(cluster_id + 1)
        return span_starts, span_ends, mention_span, cluster_ids

class wordLevelDocument(Document):
    """
        Dataset for "Word-Level Coreference Resolution" 2021 EMNLP
        Reference code: https://github.com/vdobrovolskii/wl-coref
    """
    def __init__(self, raw_text, tokens, sents, corefs, speakers, genre, filename, entities):
        super().__init__(raw_text, tokens, sents, corefs, speakers, genre, filename, entities)
        
        self.document_id = self.filename
        
        self.cased_words, self.sent_id = self._get_word_sent_map()

        # TODO: Check what part_id stands for
        self.part_id = 0
        
        self.pos, self.deprel, self.head = self._syntax_parser()
        
        self.span_clusters = self._compute_span_clusters()
        self.head2span, self.word_clusters = self._compute_word_clusters()

        # tokenize
        self.word2subword, self.subwords, self.word_id = self._tokenize()

    def _get_word_sent_map(self):
        word_sent_map = [(word, sent_idx) for sent_idx, sent in enumerate(self.sents) for word in sent]
        cased_words = [item[0] for item in word_sent_map]
        sent_ids = [item[1] for item in word_sent_map]
        
        return cased_words, sent_ids
    
    def _syntax_parser(self):
        """
            Generate POS, Dependency Relationships, Head words by Stanza
            https://stanfordnlp.github.io/stanza/depparse.html
        """
        global syntax_parser

        doc_pos = []
        doc_deprel = []
        head_words = []

        sent_start_offset = 0
        for sent in self.sents:
            parsed_text = syntax_parser([sent])

            for parsed_sent in parsed_text.sentences:
                for parsed_word in parsed_sent.words:
                    doc_pos.append(parsed_word.pos)
                    doc_deprel.append(parsed_word.deprel)

                    if parsed_word.head > 0:
                        head_words.append(parsed_word.head-1+sent_start_offset)
                    else:
                        head_words.append(None)
            sent_start_offset += len(sent)

        return doc_pos, doc_deprel, head_words
    
    def _compute_span_clusters(self):
        
        coref_cluster_num = max([coref['label'] for coref in self.corefs])+1
        span_clusters = [[] for _ in range(coref_cluster_num)]

        for coref in self.corefs:
            # wordLevelModel中 mention = (real_start, real_end+1)
            span_clusters[coref['label']].append([coref['span'][0], coref['span'][1]+1])
        
        return span_clusters
    
    def _compute_word_clusters(self):
        """
            Select head word for each mention
            return head2span, head_clusters
        """
        # get head_word for mention
        head2span = []
        for mention in [item for cluster in self.span_clusters for item in cluster]:
            head_of_mention = self._get_head(mention)
            head2span.append([head_of_mention, mention[0], mention[1]])

        head_clusters = [[self._get_head(mention) for mention in cluster] for cluster in self.span_clusters]

        return head2span, head_clusters

    def _get_head(self, mention):
        """
            Returns the span's head, which is defined as the only word within the
            span whose head is outside of the span or None. In case there are no or
            several such words, the rightmost word is returned

            Args:
                mention (Tuple[int, int]): start and end (exclusive) of a span

            Returns:
                int: word id of the spans' head
        """
        head_candidates = set()
        start, end = mention
        for i in range(start, end):
            ith_head = self.head[i]
            if ith_head is None or not (start <= ith_head < end):
                head_candidates.add(i)
        if len(head_candidates) == 1:
            return head_candidates.pop()
        return end - 1
    
    def _tokenize(self):
        """
            Use Pre-trained Model to tokenize the doc
        """
        global bert_tokenizer
        self.span_clusters = [[tuple(mention) for mention in cluster] for cluster in self.span_clusters]
        
        word2subword = []
        subwords = []
        word_id = []
        for i, word in enumerate(self.cased_words):
            tokenized_word = bert_tokenizer.tokenize(word)
            # tokenized_word = list(filter(filter_func, tokenized_word))
            word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
            subwords.extend(tokenized_word)
            word_id.extend([i] * len(tokenized_word))

        return word2subword, subwords, word_id

class CyberDocument(Document):
    def __init__(self, raw_text, tokens, sents, corefs, speakers, genre, filename, entities):
        super().__init__(raw_text, tokens, sents, corefs, speakers, genre, filename, entities)

        self.bert_tokens = []
        self.word_subtoken_map = [] # nltk_token2bert_token
        self.word2idx = {}
        self.segment_subtoken_map = []
        self.segments = []
        
        self.sentence_ends, self.token_ends, self.sentence_ends_subtok = [], [], []
        self.word2sent = []
        self.sent2subtok_bdry = []

        self.pos, self.deprel = self._syntax_parser()

        if args.bert_based and args.model != 'corefQAModel':
            self._compute_bert_tokenization()
        else:
            raise ValueError('You should choose a different Document class or you should revise the configs.')
    
    def _syntax_parser(self):
        """
            Generate POS, Dependency Relationships, Head words by Stanza
            https://stanfordnlp.github.io/stanza/depparse.html
        """
        global syntax_parser

        doc_pos = []
        doc_deprel = []

        sent_start_offset = 0
        for sent in self.sents:
            parsed_text = syntax_parser([sent])

            for parsed_sent in parsed_text.sentences:
                for parsed_word in parsed_sent.words:
                    doc_pos.append(parsed_word.pos)
                    doc_deprel.append(parsed_word.deprel)

            sent_start_offset += len(sent)

        return doc_pos, doc_deprel

    def _compute_bert_tokenization(self):
        """
        生成bert_sub_tokens
        
        self.bert_tokens: List, 将nltk分词后生成token重新进行tokenize, 转换为bert的token字典中的索引
        self.word_subtoken_map: List, size:len(bert_tokens), 每个bert_token对应的word
        self.word2subtok: Dict, key:token_idx values:sub_token_idx, token到sub_token的转换
        self.segments: List, size:segment_num, item:Tensor, 将文章分为若干小于max_segment_len的segment
        self.segment_subtoken_map: List, size:len(subtokens)+2*segment_num, 在word_subtoken_map基础上加入[CLS][SEP]位置
        """
        global bert_tokenizer
        
        max_segment_len = int(args.max_segment_len)
        
        idx_word = 0
        idx_subtoken = 0
        
        for j, sent in enumerate(self.sents):
            for i, token in enumerate(sent):
                
                subtokens = bert_tokenizer(token)['input_ids'][1:-1]
                
                self.bert_tokens.extend(subtokens)
                
                self.token_ends.extend([False] * len(subtokens))
                self.token_ends[-1] = True

                self.sentence_ends.extend([False] * len(subtokens))
                
                self.word_subtoken_map.extend([idx_word] * len(subtokens))
                self.word2sent.append(j)
                idx_word += 1
                idx_subtoken += len(subtokens)
                
            self.sentence_ends[-1] = True
        
        # 将doc按照max_segment_len进行分割，保证分割部分为句的结束
        current = 0
        previous_token = 0
        
        while current < len(self.bert_tokens):
            # Min of last token of the document, or 
            # - 2 to include [CLS] and [SEP] tokens, -1 to refer to the corrent arr element
            end = min(current + max_segment_len - 1 - 2, len(self.bert_tokens) - 1)

            while end >= current and not self.sentence_ends[end]:
                # make end of segment be end of some sentence, or equal to current token
                end -= 1
            # How can end be less than current? Only if it is less by 1 (previous constraint not satisfied)
            if end < current:
                # Put the end token back?
                end = min(current + max_segment_len - 1 - 2,  len(self.bert_tokens) - 1)
                assert self.word2sent[self.word_subtoken_map[current]] == self.word2sent[self.word_subtoken_map[end]]
                # Make the end be end of last token
                while end >= current and not self.token_ends[end]:
                    end -= 1
                if end < current:
                    raise Exception("Can't find valid segment")
                    
            # Make segment consist of subtokens for found boundaries
            self.segments.append(torch.LongTensor([101] + self.bert_tokens[current : end + 1] + [102]))
            
            subtoken_map = self.word_subtoken_map[current : end + 1]
            
            # Make the [CLS] token of the segment map to last word of previous segment and [SEP] token
            # to last word in the current segment.
            self.segment_subtoken_map.extend([previous_token] + subtoken_map + [subtoken_map[-1]])
            
            subtoken_sent_ends = self.sentence_ends[current : end + 1]
            subtoken_sent_ends[-1] = False
            self.sentence_ends_subtok.extend([False] + subtoken_sent_ends + [True])
                
            current = end + 1
            previous_token = subtoken_map[-1]
            
            
        self.word2subtok = defaultdict(list)
        sentence_idx = 0
        for i, word_idx in enumerate(self.segment_subtoken_map):
            self.word2subtok[word_idx].append(i)
            # If current token is an end of sentence
            if self.sentence_ends_subtok[i]:
                self.sent2subtok_bdry.append((sentence_idx, i))
                sentence_idx = i+1


    def truncate(self):
        """ 
        子类重构自裁
        Randomly truncate the document to up to MAX sentences 
        """
        MAX = args.segment_max_num
        if len(self.segments) > MAX: 
            # 以i作为终止，随机选择MAX个segments进行训练
            i = random.sample(range(MAX, len(self.segments)), 1)[0]
            subtokens = flatten(self.segments[i-MAX:i])
            
            
            # Index of the first token in the truncated segments
            num_pre_subtokens = len(flatten(self.segments[0:i-MAX]))
            # Index of the last token in the truncated segments
            num_pre_curr_subtokens = num_pre_subtokens + len(subtokens) -1
            
            # Index of the first and the last word corresponding to 
            # given truncated segments
            first_word_idx, last_word_idx = self.segment_subtoken_map[num_pre_subtokens], \
                                                self.segment_subtoken_map[num_pre_curr_subtokens]
            
            first_sentence_idx, last_sentence_idx = self.word2sent[first_word_idx], \
                                                        self.word2sent[last_word_idx]
            sents = self.sents[first_sentence_idx:last_sentence_idx + 1]
                # +1 to include last sentence too
            tokens = flatten(sents)
            pre_sents = self.sents[0:first_sentence_idx]
            pre_tokens = flatten(pre_sents)
            # Index of first token in truncated sentences
            num_pre_tokens = len(pre_tokens)
            # Index of last token in truncated sentences
            num_pre_curr_tokens = num_pre_tokens + len(tokens) - 1
            new_corefs = []

            for coref in self.corefs:
                # Ignore corefs outside of current sentences
                if coref['start'] < num_pre_tokens or coref['end'] > num_pre_curr_tokens:
                    continue
                new_start = coref['start'] - num_pre_tokens
                new_end = coref['end'] - num_pre_tokens
                new_coref = {'label': coref['label'], 
                                'start': new_start,
                                'end': new_end,
                                'span': (new_start, new_end)}
                new_corefs.append(new_coref)

            new_speakers = self.speakers[num_pre_tokens:num_pre_curr_tokens + 1]

            return self.__class__(c(self.raw_text), tokens, sents,
                                    new_corefs, new_speakers,
                                    c(self.genre), c(self.filename), c(self.entities))
        return self


@attr.s(frozen=True, repr=False)
class Span:

    # Left / right token indexes
    i1 = attr.ib()
    i2 = attr.ib()

    # Id within total spans (for indexing into a batch computation)
    id = attr.ib()

    # Speaker
    speaker = attr.ib(default=None)

    # Genre
    genre = attr.ib(default=None)

    # Unary mention score, as tensor
    si = attr.ib(default=None)

    # List of candidate antecedent spans
    yi = attr.ib(default=[])

    # Corresponding span ids to each yi
    yi_idx = attr.ib(default=None)

    # the content of span
    content = attr.ib(default=None)

    # the pred type of span
    type = attr.ib(default=None)

    # the type_embed of span
    type_embed = attr.ib(default=None)

    # the sent_if of current document
    sent_id = attr.ib(default=None)

    def __len__(self):
        return self.i2-self.i1+1

    def __repr__(self):
        return 'Span representing %d tokens' % (self.__len__())


def load_ann_file(filepath):
    """
    Load a ann format file
    Input:
        filepath: path to a ann file (make sure the path contain the corresponding txt file)
    Output:
        a_document: a instance of the Document class which containing:
            tokens:                   split list of text
            utts_corefs:
                coref['label']:     id of the coreference cluster
                coref['start']:     start index (index of first token in the utterance)
                coref['end':        end index (index of last token in the utterance)
                coref['span']:      corresponding span
            utts_speakers:          list of speakers (default to be speaker#1)
            genre:                  genre of input (default to be sec)
    """

    txt_file_path = filepath.rstrip('.ann') + '.txt'
    with open(txt_file_path,'r') as txt_file:
        raw_text = txt_file.read()
        
    # 分句
    sents = sent_tokenize(raw_text)
    sents = [tokenize(sent, offset=False) for sent in sents]

    # 分词
    tokens, tokens_offset = tokenize(raw_text)
    # 实体
    entities = read_entity(filepath)
    # 将token对应到实体上
    for i, entity in enumerate(entities):
        entity_offset_from = int(entity[2])
        entity_offset_to = int(entity[3])

        matched_tokens = []
    
        for j, token_offset in enumerate(tokens_offset):
            if token_offset[0] >= entity_offset_to or token_offset[1] <= entity_offset_from:
                continue
            elif token_offset[0]>=entity_offset_from and token_offset[1]<=entity_offset_to:
                matched_tokens.append(j)
            else:
                print("Error! token offsets does not aligned with entity offsets!")
                print('Filepath: %s' % filepath)
                print("Entity: %s" % entity[4])
                print("Token: %s\n" % raw_text[token_offset[0]:token_offset[1]])

                with open('./Dataset/error_annotation_file.txt', 'a') as error_annotation_file:
                    error_annotation_file.write('Filepath: %s\n' % filepath)
                    error_annotation_file.write("Entity: %s\n" % entity[4])
                    error_annotation_file.write("Token: %s\n\n" % raw_text[token_offset[0]:token_offset[1]])

        entities[i].append(matched_tokens)
    
    # 共指关系簇
    utts_corefs = []
    coref_relations = read_coref(filepath)
    for index, coref_cluster in enumerate(coref_relations):
        for coref_entity in coref_cluster:
            for entity in entities:
                if entity[0] == coref_entity:
                    entity[5].sort()
                    start = entity[5][0]
                    end = entity[5][-1]

            utts_corefs.append(
                {
                    'label': index,
                    'start': start,
                    'end' : end,
                    'span': (start, end)
                }
            )

    # 为每个分词生成默认的speaker
    utts_speakers = ['-'] * len(tokens)
    # 文章类别
    genre = 'sec'
    # 文件名
    filename = filepath.split('/')[-1].split('.')[0]
    
    if args.model == 'wordLevelModel':
        doc = wordLevelDocument(raw_text, tokens, sents, utts_corefs, utts_speakers, genre, filename, entities) 
    elif args.model == 'corefQAModel':
        doc = corefQADocument(raw_text, tokens, sents, utts_corefs, utts_speakers, genre, filename, entities)
    elif args.model == 'cyberCorefModel':
        doc = CyberDocument(raw_text, tokens, sents, utts_corefs, utts_speakers, genre, filename, entities)
    elif args.bert_based:
        doc = BERTDocument(raw_text, tokens, sents, utts_corefs, utts_speakers, genre, filename, entities)
    else:
        doc = Document(raw_text, tokens, sents, utts_corefs, utts_speakers, genre, filename, entities)
    
    return doc

def tokenize(raw_text, offset=True):
    """
    对文本进行分词
    基本要求是 token的边界应于实体的边界对齐, 即token必须完全属于某一已标注的实体或者完全不属于

    TODO: 首先标注出实体边界不对齐的文件, 手动校队; 在确诊标注之后, 针对e.g. 2Wirese等粘连情况自动按照实体标注界限分割Token
    """
    # 细粒度分割，首先强制分割所有标点符号
    punkt_split_text = re.split(r"([.,!?;:,+\'\"\“”@()$~£\\])", raw_text)
    
    # 相对每个子句的offset
    punkt_split_sentence_offset = [list(WordPunctTokenizer().span_tokenize(sentence)) for sentence in punkt_split_text]

    # 将所有相对子句的offset还原成相对raw_text的offset
    current_offset = 0
    spans_offset = []
    for i, subsent_offset in enumerate(punkt_split_sentence_offset):
        for span_offset in subsent_offset:
            spans_offset.append((span_offset[0]+current_offset, span_offset[1]+current_offset))
        
        current_offset += len(punkt_split_text[i])

    tokens = [raw_text[token_offset[0]:token_offset[1]] for token_offset in spans_offset]

    if offset == True:
        return tokens, spans_offset
    else:
        return tokens

def read_entity(ann_fname):
    """
    根据ann文件路径读取实体信息
    return entity_info = [[e_index, entity_type, from, to, content], ...]   
    """
    entity_info = []

    for line in open(ann_fname, 'r'):
        # 如果不是T开头 说明已经标了Relation
        if not line.startswith('T'):
            continue
        
        index = re.search(r'^\w\d*', line).group(0)
        e_type = re.search(r'(?<=\t)[\w-]*', line).group(0)
        e_from = re.search(r'(?<=\s)[0-9]{1,}',line).group(0)
        e_to = re.findall(r'(?<=\s)[0-9]{1,}',line)[1]
        e_content = line.split('\t')[-1].strip('\n')

        entity_info.append([index, e_type, e_from, e_to, e_content])

    return entity_info

def read_coref(ann_fname):
    """
    读取ann文件中的coref关系
    return : coref_relations = [[T1, T2, ...], [T4, T5, ...]]
    """
    coref_relations = []

    for line in open(ann_fname, 'r'):
        if not line.startswith('*'):
            continue

        elif re.search(r'(?<=\t)[\w-]*', line).group(0) != 'Coreference':
            continue

        else:
            line = line.rstrip('\n')
            coref_entities = line.split('\t')[1].split(' ')[1:]
            coref_relations.append(coref_entities)

    return coref_relations

def load_corpus(corpus_dir):
    corpus = pickle.load(open(corpus_dir, 'rb'))
    return corpus

def generate_casie_corpus(rootdir, ratio):
    """
    从ann的手动标注文件生成用于可用于训练的dataset
    Generate dataset for training from annotation files in brat format
    Input:
        rootdir: the directory where saves annotation files
        ratio: the ratio of raw data to generate training dataset, default=1.0
    """

    filenames = [filename for filename in os.listdir(rootdir) if filename.endswith('.ann')]

    train_filenames, valset_filenames = dataset_split(filenames, args.train_val_ratio, False)
    
    # Make sure every time select the same data item
    random.seed(0)
    train_filenames = random.sample(train_filenames, k=int(len(train_filenames)*ratio))
    valset_filenames = random.sample(valset_filenames, k=int(len(valset_filenames)*ratio))

    train_docs = []
    for train_filename in tqdm(train_filenames):
        try:
            train_doc = load_ann_file(rootdir+train_filename)
            train_docs.append(train_doc)
        except(IndexError):
            print("IndexError when dealing with: %s\n" % train_filename)
        except(LookupError):
            nltk.download('punkt')
            train_doc = load_ann_file(rootdir+train_filename)
            train_docs.append(train_doc)
        except:
            print("Error when dealing with %s\n" % train_filename)

    train_corpus = Corpus(train_docs)

    train_save_path = args.dataset_path + args.corpus_subpath + '/train' + args.corpus_filename
    pickle.dump(train_corpus, open(train_save_path, 'wb'))
    print("train_corpus docs: num", len(train_corpus.docs))

    val_docs = []
    for val_filename in tqdm(valset_filenames):
        try:
            val_doc = load_ann_file(rootdir+val_filename)
            val_docs.append(val_doc)
        except:
            print("Error when dealing with %s\n" % val_filename)
    val_corpus = Corpus(val_docs)

    val_save_path = args.dataset_path + args.corpus_subpath + '/val' + args.corpus_filename
    pickle.dump(val_corpus, open(val_save_path, 'wb'))
    print("val_corpus docs: num", len(val_corpus.docs))

def dataset_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio随机划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_2, sublist_1


if __name__ == "__main__":
    """
    Example for generate corpus:
    + bert based:
    python dataLoader.py --model bertCorefModel --bert_based --bert_name bert-base --max_segment_len 384 --corpus_subpath casieAll_0430 --corpus_filename _corpus_bert_base.pkl --train_val_ratio 0.2
    
    + span-bert based:
    python dataLoader.py --model bertCorefModel --bert_based --bert_name spanbert-base --max_segment_len 384 --corpus_subpath casieAll_0430 --corpus_filename _corpus_spanbert_base.pkl
    
    # python dataLoader.py --model bertCorefModel --bert_based --bert_name spanbert-base --max_segment_len 384 --corpus_subpath casie100_10_0430 --corpus_filename _corpus_spanbert_base.pkl --train_val_ratio 0.1

    python dataLoader.py --model cyberCorefModel --bert_based --bert_name spanbert-base --max_segment_len 384 --corpus_subpath casieAll_0430 --corpus_filename _corpus_cyber.pkl --train_val_ratio 0.2
    
    + coref-bert based:
    python dataLoader.py --model bertCorefModel --bert_based --bert_name corefbert-base --max_segment_len 384 --corpus_subpath casieAll_0430 --corpus_filename _corpus_corefbert_base.pkl --train_val_ratio 0.2
    
    + neural networks based:
    python dataLoader.py --model nnCorefModel --sentense_max_num 50 --corpus_subpath casieAll_0430 --corpus_filename _corpus_nn.pkl --train_val_ratio 0.2

    + wordLevelModel
    python dataLoader.py --model wordLevelModel --bert_based --bert_name spanbert-base --corpus_subpath casieAll_0430 --corpus_filename _corpus_wordLevel.pkl --train_val_ratio 0.2
    """

    generate_casie_corpus("./Dataset/rawData/CasieCoref0430/", ratio=1)

    # train_corpus_path = args.dataset_path + args.corpus_subpath + '/train' + args.corpus_filename
    # val_corpus_path = args.dataset_path + args.corpus_subpath + '/val' + args.corpus_filename
    # train_corpus = load_corpus(train_corpus_path)
    # val_corpus = load_corpus(val_corpus_path)

    # span_length = []
    # for doc in train_corpus:
    #     for coref in doc.corefs:
    #         span_length.append(coref['span'][1]-coref['span'][0]+1)
    # for doc in val_corpus:
    #     for coref in doc.corefs:
    #         span_length.append(coref['span'][1]-coref['span'][0]+1)
    
    # result = Counter(span_length)
    # print(result)