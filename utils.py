import os
import io
from re import L
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, pack_sequence

import numpy as np
from boltons.iterutils import pairwise, windowed
from itertools import groupby, combinations
from collections import defaultdict

import itertools
from scipy.optimize import linear_sum_assignment

from config import arg_parse

args = arg_parse()

EPSILON = 1e-7

def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return to_cuda(x).requires_grad_()

def to_cuda(x, cuda_id=0):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda(cuda_id)
    return x

def unpack_and_unpad(lstm_out, reorder):
    """ Given a padded and packed sequence and its reordering indexes,
    unpack and unpad it. Inverse of pad_and_pack """

    # Restore a packed sequence to its padded version
    unpacked, sizes = pad_packed_sequence(lstm_out, batch_first=True)

    # Restored a packed sequence to its original, unequal sized tensors
    unpadded = [unpacked[idx][:val] for idx, val in enumerate(sizes)]

    # Restore original ordering
    regrouped = [unpadded[idx] for idx in reorder]

    return regrouped

def pad_and_stack(tensors, pad_size=None, value=0):
    """ Pad and stack an uneven tensor of token lookup ids.
    Assumes num_sents in first dimension (batch_first=True)"""

    # Get their original sizes (measured in number of tokens)
    sizes = [s.shape[0] for s in tensors]

    # Pad size will be the max of the sizes
    if not pad_size:
        pad_size = max(sizes)

    # Pad all sentences to the max observed size
    # TODO: why does pad_sequence blow up backprop time? (copy vs. slice issue)
    padded = torch.stack([F.pad(input=sent[:pad_size],
                                pad=(0, 0, 0, max(0, pad_size-size)),
                                value=value)
                          for sent, size in zip(tensors, sizes)], dim=0)

    return padded

def pack(tensors):
    """ Pack list of tensors, provide reorder indexes """

    # Get sizes
    sizes = [t.shape[0] for t in tensors]

    # Get indexes for sorted sizes (largest to smallest)
    size_sort = np.argsort(sizes)[::-1]

    # Resort the tensor accordingly
    sorted_tensors = [tensors[i] for i in size_sort]

    # Resort sizes in descending order
    sizes = sorted(sizes, reverse=True)

    # Pack the padded sequences
    packed = pack_sequence(sorted_tensors)

    # Regroup indexes for restoring tensor to its original order
    reorder = torch.tensor(np.argsort(size_sort), requires_grad=False)

    return packed, reorder

def prune(spans, T, LAMBDA=args.prune_lambda):
    """ Prune mention scores to the top lambda percent.
    Returns list of tuple(scores, indices, g_i) """

    # Only take top λT spans, where T = len(doc)
    STOP = int(LAMBDA * T)

    # Sort by mention score, remove overlapping spans, prune to top λT spans
    sorted_spans = sorted(spans, key=lambda s: s.si, reverse=True)
    nonoverlapping = remove_overlapping(sorted_spans)
    pruned_spans = nonoverlapping[:STOP]

    # Resort by start, end indexes
    spans = sorted(pruned_spans, key=lambda s: (s.i1, s.i2))

    return spans

def remove_overlapping(sorted_spans):
    """ Remove spans that are overlapping by order of decreasing mention score
    unless the current span i yields true to the following condition with any
    previously accepted span j:
    si.i1 < sj.i1 <= si.i2 < sj.i2   OR
    sj.i1 < si.i1 <= sj.i2 < si.i2 """

    # Nonoverlapping will be accepted spans, seen is start, end indexes that
    # have already been seen in an accepted span
    nonoverlapping, seen = [], set()
    for s in sorted_spans:
        indexes = range(s.i1, s.i2+1)
        taken = [i in seen for i in indexes]
        if len(set(taken)) == 1 or (taken[0] == taken[-1] == False):
            nonoverlapping.append(s)
            seen.update(indexes)

    return nonoverlapping

def prune_bert(mention_scores, start_words, end_words, doc, LAMBDA=args.prune_lambda):
    #Prune mention scores to the top lambda percent.
    #Returns list of tuple(scores, indices, g_i) 

    # Only take top λT spans, where T = len(doc)
    T = len(doc.tokens)
    STOP = int(LAMBDA * T)

    # Expriments for coref resolution with all gold mentions
    # STOP = len(doc.corefs)
    
    # check_nested_indices = check_nested_indices_1(start_words, end_words, doc_words)
    # Sort by mention score, remove overlapping spans, prune to top λT spans
    indices = torch.argsort(mention_scores.view(-1), descending=True)
    # checked_nested_indices = check_nested_indices(torch.tensor(start_words)[indices], 
    #                                             torch.tensor(end_words)[indices], 
    #                                             indices,
    #                                             doc_words)
    nonoverlapping_indices = remove_overlapping_bert(torch.tensor(start_words)[indices], 
                                                torch.tensor(end_words)[indices], 
                                                indices, 
                                                STOP)
    
    # Resort by start, end indexes
    indices_sorted = sorted(nonoverlapping_indices, key=lambda i: (start_words[i], end_words[i]))

    return indices_sorted

def check_nested_indices_1(start_words, end_words, indices, words):
    # Only keep pronoun article prefix check
    # For example:

    # Given two spans: the biggest hacker group, biggest hacker group
    # only keeps the biggest hacker group

    prefix_dict = {
        'Artical': ['a', 'A', 'an', 'An', 'the', 'The'],
        'Pronoun': ['my', 'My', 'his', 'its', 'His', 'Its', 'your', 'hers', 'ours', 'this', 'Your', 'Hers', 'Ours', 'This', 'their', 'these', 'those', 'Their', 'These', 'Those']
    }

    articals_words, pronoun_words = [], []
    for i, word in enumerate(words):
        if word in prefix_dict['Artical']:
            articals_words.append(i)
        elif word in prefix_dict['Pronoun']:
            pronoun_words.append(i)

    removed_indicies_list = []

    start_words_len_map = []
    cur_start_word = start_words[0]
    cur_count = 0
    for i in range(len(start_words)):
        if start_words[i] == cur_start_word:
            cur_count += 1
        else:
            start_words_len_map.append(cur_count)
            cur_start_word = start_words[i]
            cur_count = 1
    start_words_len_map.append(cur_count)

    i = 0
    splited_start_words = []
    splited_end_words = []
    for length in start_words_len_map:
        splited_start_words.append(start_words[i:i+length])
        splited_end_words.append(end_words[i:i+length])
        i += length

    

    ## TODO: O(T^4) which is unacceptable
    for i in range(len(start_words)):
        if start_words[i] in articals_words or start_words[i] in pronoun_words:
            for j in range(len(start_words)):
                if start_words[i] == start_words[j] + 1 and end_words[i] == end_words[j]:
                    removed_indicies_list.append(j)

    keeped_indicies_list = list(set(range(len(start_words))) - set(removed_indicies_list))
    
    return indices[keeped_indicies_list]


def check_nested_indices(start_words, end_words, indices, words):
    # Only keep pronoun article prefix check
    # For example:

    # Given two spans: the biggest hacker group, biggest hacker group
    # only keeps the biggest hacker group

    prefix_dict = {
        'Artical': ['a', 'A', 'an', 'An', 'the', 'The'],
        'Pronoun': ['my', 'My', 'his', 'its', 'His', 'Its', 'your', 'hers', 'ours', 'this', 'Your', 'Hers', 'Ours', 'This', 'their', 'these', 'those', 'Their', 'These', 'Those']
    }

    articals_words, pronoun_words = [], []
    for i, word in enumerate(words):
        if word in prefix_dict['Artical']:
            articals_words.append(i)
        elif word in prefix_dict['Pronoun']:
            pronoun_words.append(i)

    removed_indicies_list = []

    ## TODO: O(T^4) which is unacceptable
    for i in range(len(start_words)):
        if start_words[i] in articals_words or start_words[i] in pronoun_words:
            for j in range(len(start_words)):
                if start_words[i] == start_words[j] + 1 and end_words[i] == end_words[j]:
                    removed_indicies_list.append(j)

    keeped_indicies_list = list(set(range(len(start_words))) - set(removed_indicies_list))
    
    return indices[keeped_indicies_list]

def remove_overlapping_bert(start_words, end_words, indices, STOP):
    #Remove spans that are overlapping by order of decreasing mention score
    #unless the current span i yields true to the following condition with any
    #previously accepted span j:

    #si.i1 < sj.i1 <= si.i2 < sj.i2   OR
    #sj.i1 < si.i1 <= sj.i2 < si.i2

    # TODO: Pretty brute force (O(n^2)), rewrite it later
    nonoverlapping_indices, overlapped = [], False
    for i in range(len(start_words)):        
        for j in nonoverlapping_indices:
            if (start_words[i] < start_words[j] and start_words[j] <= end_words[i] and end_words[i] < end_words[j] or \
                    start_words[j] < start_words[i] and start_words[i] <= end_words[j] and end_words[j] < end_words[i]):
                
                overlapped = True
                break
                
        if not overlapped:
            nonoverlapping_indices.append(i)
                
        overlapped = False
                
        if len(nonoverlapping_indices) == STOP:
            break
        
    return indices[nonoverlapping_indices]

    
def pairwise_indexes(spans):
    """ Get indices for indexing into pairwise_scores """
    indexes = [0] + [len(s.yi) for s in spans]
    indexes = [sum(indexes[:idx+1]) for idx, _ in enumerate(indexes)]
    return pairwise(indexes)


def extract_gold_corefs(document):
    """ Parse coreference dictionary of a document to get coref links """

    # Initialize defaultdict for keeping track of corefs
    gold_links = defaultdict(list)

    # Compute number of mentions
    gold_mentions = set([coref['span'] for coref in document.corefs])
    total_mentions = len(gold_mentions)
    # Compute number of coreferences
    for coref_entry in document.corefs:

        # Parse label of coref span, the span itself
        label, span_idx = coref_entry['label'], coref_entry['span']

        # All spans corresponding to the same label
        gold_links[label].append(span_idx) # get all spans corresponding to some label
    # Flatten all possible corefs, sort, get number
    gold_corefs = flatten([[coref
                            for coref in combinations(gold, 2)]
                            for gold in gold_links.values()])
    
    gold_corefs = sorted(gold_corefs)
    total_corefs = len(gold_corefs)
    
    return gold_corefs, total_corefs, gold_mentions, total_mentions


def compute_idx_spans_for_bert(sentences, L, word2tokens=None):
    # Compute span indexes for all possible spans up to length L in each
    #sentence 
    shift = 0    
    start_words, end_words, start_toks, end_toks, tok_ranges, word_widths, tok_widths, sent_ids = [], [], [], [], [], [], [], []
    for sent_id, sent in enumerate(sentences):
        # sent_spans = []
        for length in range(1, min(L, len(sent))):
            l_spans = windowed(range(shift, len(sent)+shift), length)
            try:
                flattened = flatten_word2tokens(l_spans, word2tokens)
            except(IndexError):
                print(sentences)
                print(length)
                print(l_spans)
                print(word2tokens)
            start_words.extend(flattened[0])
            end_words.extend(flattened[1])
            start_toks.extend(flattened[2])
            end_toks.extend(flattened[3])
            tok_ranges.extend(flattened[4])
            word_widths.extend(flattened[5])
            tok_widths.extend(flattened[6])
            sent_ids.extend([sent_id]*len(flattened[0]))
        shift += len(sent)
    return start_words, end_words, start_toks, end_toks, tok_ranges, word_widths, tok_widths, sent_ids

def compute_idx_spans(sentences, L=args.max_span_length):
    """ Compute span indexes for all possible spans up to length L in each
    sentence """
    idx_spans, shift = [], 0
    for sent in sentences:
        sent_spans = flatten([windowed(range(shift, len(sent)+shift), length)
                              for length in range(1, L)])
        idx_spans.extend(sent_spans)
        shift += len(sent)

    return idx_spans


def add_dummy(tensor: torch.Tensor, eps: bool = False):
    """ 
        "Word Level Coreference Resolution"
        Prepends zeros (or a very small value if eps is True)
        to the first (not zeroth) dimension of tensor.
    """
    kwargs = dict(device=tensor.device, dtype=tensor.dtype)
    shape = list(tensor.shape)
    shape[1] = 1
    if not eps:
        dummy = torch.zeros(shape, **kwargs)          # type: ignore
    else:
        dummy = torch.full(shape, EPSILON, **kwargs)  # type: ignore
    return torch.cat((dummy, tensor), dim=1)


def extract_gold_coref_cluster(document):
    """
    获取document内的共指簇
    Input: Document
    Output: Cluster [((1,2), (7,8), (9,10)), ...]
    """

    coref_cluster = {}
    for coref in document.corefs:
        if coref['label'] not in coref_cluster.keys():
            coref_cluster[coref['label']] = []
        
        coref_cluster[coref['label']].append(coref['span'])
    
    return list(coref_cluster.values())

def extract_pred_coref_cluster(spans, scores):
    """
    根据spans获取共指簇
    Input: spans->[span, span, ...]  item: Span]
           scores->Tensor(mention_num, K+1)
    Output: Cluster [((1,2), (7,8), (9,10)), ...]
    """

    coref_cluster = []
    for i, span in enumerate(spans):
        # 跳过第一个span
        if i == 0:
            continue
        cur_span = (span.i1, span.i2)
        
        found_coref = torch.argmax(scores[i, :])
        if scores[i, found_coref] > 0:
            candi_span = spans[i].yi[found_coref-1]
            candi_span = (candi_span.i1, candi_span.i2)
        else:
            continue

        # find the cluster
        find_flag = False
        for i, cur_cluster in enumerate(coref_cluster):
            if cur_span in cur_cluster or candi_span in cur_cluster:
                cur_cluster.append(cur_span)
                cur_cluster.append(candi_span)

                coref_cluster[i] = list(set(cur_cluster))
                find_flag = True
        
        if not find_flag:
            coref_cluster.append([cur_span, candi_span])

    return coref_cluster

def s_to_speaker(span, speakers):
    """ Compute speaker of a span """
    if speakers[span.i1] == speakers[span.i2]:
        return speakers[span.i1]
    return None

def speaker_label(s1, s2):
    """ Compute if two spans have the same speaker or not """
    # Same speaker
    if s1.speaker == s2.speaker:
        idx = torch.tensor(1)

    # Different speakers
    elif s1.speaker != s2.speaker:
        idx = torch.tensor(2)

    # No speaker
    else:
        idx = torch.tensor(0)

    return to_cuda(idx)

def safe_divide(x, y):
    """ Make sure we don't divide by 0 """
    if y != 0:
        return x / y
    return 0.0

def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]

def flatten_word2tokens(alist, word2tokens):
    """ Flatten a list of lists into one list """
    start_words, end_words, start_toks, \
            end_toks, tok_ranges, word_widths, \
            tok_widths = zip(*[(window[0], 
                                window[-1], 
                                word2tokens[window[0]][0], 
                                word2tokens[window[-1]][-1],
                                list(range(word2tokens[window[0]][0], word2tokens[window[-1]][-1] + 1)),
                                window[-1] - window[0] + 1,
                                word2tokens[window[-1]][-1] + 1 - word2tokens[window[0]][0])
                            for window in alist])
    
    return  start_words, end_words, start_toks, \
            end_toks, flatten(tok_ranges), word_widths, tok_widths

def get_f1(precision, recall):
    """
    模型训练算法个人部分
    Parameters
    ------
        precision       float       准确率
        recall          float       召回率
    Return
    ------
        float       调和平均数
    """
    return safe_divide(precision * recall * 2, (precision + recall))


def muc_old(predicted_clusters, gold_clusters):
    """
    the link based MUC
    Parameters
    ------
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    Return
    ------
        tuple(float)    准确率、召回率、调和平均数
    """
    pred_edges = set()
    for cluster in predicted_clusters:
        pred_edges |= set(itertools.combinations(cluster, 2))
    gold_edges = set()
    for cluster in gold_clusters:
        gold_edges |= set(itertools.combinations(cluster, 2))
    correct_edges = gold_edges & pred_edges
    precision = safe_divide(len(correct_edges), len(pred_edges))
    recall = safe_divide(len(correct_edges), len(gold_edges))
    f1 = get_f1(precision, recall)
    return precision, recall, f1


def muc(predicted_clusters, gold_clusters):
    """
    MUC-6
    Parameters
    ------
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    Return
    ------
        tuple(float)    准确率、召回率、调和平均数
    """

    total_gold_cluster_len = []
    total_partition_gold_cluster_len = []

    gold_mentions = [item for cluster in gold_clusters for item in cluster]
    pred_mentions = [item for cluster in predicted_clusters for item in cluster]

    for gold_cluster in gold_clusters:

        # 首先对于不在pred_clusters中的gold_mentions默认以singleton处理，也就是论文中所提到的implicitly
        # TODO:但是这样会优化结果，因为这些mention可能都没有被预测出来，而不是被模型当作singleton处理了，所以需要增添判断或者在pred,gold中把所有single加入进来
        add_singles_pred_clusters = [[item] for item in gold_mentions if item not in pred_mentions] + predicted_clusters

        partition_gold_cluster = [set(gold_cluster)&set(pred_cluster) for pred_cluster in add_singles_pred_clusters]
        # 去除partition_gold_cluster中的所有空集
        partition_gold_cluster = [item for item in partition_gold_cluster if bool(item)]

        total_gold_cluster_len.append(len(gold_cluster))
        total_partition_gold_cluster_len.append(len(partition_gold_cluster))

    total_pred_cluster_len = []
    total_partition_pred_cluster_len = []

    for pred_cluster in predicted_clusters:
        
        # TODO: 同上
        add_singles_gold_clusters = [[item] for item in pred_mentions if item not in gold_mentions] + gold_clusters

        partition_pred_cluster = [set(gold_cluster)&set(pred_cluster) for gold_cluster in add_singles_gold_clusters]
        partition_pred_cluster = [item for item in partition_pred_cluster if bool(item)]

        total_pred_cluster_len.append(len(pred_cluster))
        total_partition_pred_cluster_len.append(len(partition_pred_cluster))

    recall = safe_divide(np.sum(np.array(total_gold_cluster_len)-np.array(total_partition_gold_cluster_len)), np.sum(np.array(total_gold_cluster_len)-np.ones(len(total_gold_cluster_len))))
    precision = safe_divide(np.sum(np.array(total_pred_cluster_len)-np.array(total_partition_pred_cluster_len)), np.sum(np.array(total_pred_cluster_len)-np.ones(len(total_pred_cluster_len))))

    f1 = get_f1(precision, recall)

    return precision, recall, f1
        


def b_cubed(predicted_clusters, gold_clusters):
    """
    B cubed metric
    模型训练算法个人部分
    Parameters
    ------
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    Return
    ------
        tuple(float)    准确率、召回率、调和平均数
    """
    mentions = set(sum(predicted_clusters, [])) & set(sum(gold_clusters, []))
    precisions = []
    recalls = []
    for mention in mentions:
        mention2predicted_cluster = [x for x in predicted_clusters if mention in x][0]
        mention2gold_cluster = [x for x in gold_clusters if mention in x][0]
        corrects = set(mention2predicted_cluster) & set(mention2gold_cluster)
        precisions.append(safe_divide(len(corrects), len(mention2predicted_cluster)))
        recalls.append(safe_divide(len(corrects), len(mention2gold_cluster)))
    precision = safe_divide(sum(precisions), len(precisions))
    recall = safe_divide(sum(recalls), len(recalls))
    f1 = get_f1(precision, recall)
    return precision, recall, f1


def ceaf_phi3(predicted_clusters, gold_clusters):
    """
    the entity based CEAF metric
    Parameters
    ------
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    Return
    ------
        tuple(float)    准确率、召回率、调和平均数
    """
    scores = np.zeros((len(predicted_clusters), len(gold_clusters)))
    for j in range(len(gold_clusters)):
        for i in range(len(predicted_clusters)):
            scores[i, j] = len(set(predicted_clusters[i]) & set(gold_clusters[j]))
    indexs = linear_sum_assignment(scores, maximize=True)
    
    max_correct_mentions = sum(
        [scores[indexs[0][i], indexs[1][i]] for i in range(indexs[0].shape[0])]
    )

    precision = safe_divide(max_correct_mentions, len(sum(predicted_clusters, [])))
    recall = safe_divide(max_correct_mentions, len(sum(gold_clusters, [])))
    f1 = get_f1(precision, recall)
    
    return precision, recall, f1

def ceaf_phi4(predicted_clusters, gold_clusters):
    """
    the entity based CEAF metric
    Parameters
    ------
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    Return
    ------
        tuple(float)    准确率、召回率、调和平均数
    """
    scores = np.zeros((len(predicted_clusters), len(gold_clusters)))
    for j in range(len(gold_clusters)):
        for i in range(len(predicted_clusters)):

            scores[i, j] = 2*len(set(predicted_clusters[i]) & set(gold_clusters[j])) / (len(predicted_clusters[i])+len(gold_clusters[j]))

    indexs = linear_sum_assignment(scores, maximize=True)
    
    max_correct_mentions = sum(
        [scores[indexs[0][i], indexs[1][i]] for i in range(indexs[0].shape[0])]
    )

    precision = safe_divide(max_correct_mentions, len(sum(predicted_clusters, [])))
    recall = safe_divide(max_correct_mentions, len(sum(gold_clusters, [])))
    f1 = get_f1(precision, recall)
    
    return precision, recall, f1

def _lea(key, response):
    """ See aclweb.org/anthology/P16-1060.pdf. """
    response_clusters = [set(cluster) for cluster in response]
    response_map = {mention: cluster
                    for cluster in response_clusters
                    for mention in cluster}
    importances = []
    resolutions = []
    for entity in key:
        size = len(entity)
        if size == 1:  # entities of size 1 are not annotated
            continue
        importances.append(size)
        correct_links = 0
        for i in range(size):
            for j in range(i + 1, size):
                correct_links += int(entity[i]
                                        in response_map.get(entity[j], {}))
        resolutions.append(correct_links / (size * (size - 1) / 2))
    res = sum(imp * res for imp, res in zip(importances, resolutions))
    weight = sum(importances)
    return res, weight

def lea(predicted_clusters, gold_clusters):
    recall, r_weight = _lea(gold_clusters, predicted_clusters)
    precision, p_weight = _lea(predicted_clusters, gold_clusters)

    doc_precision = precision / (p_weight + EPSILON)
    doc_recall = recall / (r_weight + EPSILON)
    doc_f1 = (doc_precision * doc_recall) \
        / (doc_precision + doc_recall + EPSILON) * 2

    return doc_precision, doc_recall, doc_f1

def conll_coref_f1(predicted_clusters, gold_clusters):
    """
    模型训练算法个人部分
    Parameters
    ------
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    Return
    ------
        f1    调和平均数
    """
    _, _, f1_m = muc(predicted_clusters, gold_clusters)
    _, _, f1_b = b_cubed(predicted_clusters, gold_clusters)
    _, _, f1_c = ceaf_phi3(predicted_clusters, gold_clusters)
    return (f1_m + f1_b + f1_c) / 3


if __name__ == '__main__':
    
    testcases = {
        'testcase1': ([[1,2], [3,4]], [[1,2,3,4]]),
        'testcase2': ([[1,2,3,4]], [[1,2], [3,4]]),
        'testcase3': ([[1,2,3,4]], [[1,2,3,4]]),
        'testcase4': ([[1,3]], [[1,2,3]]),
        'testcase5': ([[1,2,3], [4,5,6], [7, 8, 9]], [[2,3,4,5,7,8,10]]),
        'testcase6': ([[1,2], [3,4], [5,6], [7, 8]], [[1,2,3], [4,5,6,7]]),
        'testcase7': ([[1,2], [3,4], [5,6], [7, 8], [12]], [[1,2,3], [4,5,6,7], [12]]),
        'testcase8' : ([[2,3,4]], [[2,3], [4]]),
        'testcase9' : ([[2,3,4]], [[2,3]])
    }

    for name, testcase in testcases.items():
        precision, recall, f1 = ceaf_phi4(testcase[0], testcase[1])
        print('%s: precision=%.3f, recall=%.3f, f1=%.3f' % (name, precision, recall, f1))