#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# author: yuxian meng
# description:
# corefqa model 


import torch
import torch.nn as nn
from transformers import BertModel
# from transformers import BertPreTrainedModel, BertModel
from config import arg_parse
args = arg_parse()

class CorefQA(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()

        self.bert, _ = BertModel.from_pretrained("bert-base-cased", output_loading_info=True)
        self.device = device

        # other configs
        self.pad_idx = 0
        self.max_span_width = 10
        self.span_ratio = 0.2
        self.max_candidate_num = 30
        self.max_antecedent_num = 20
        self.sliding_window_size = 128
        self.mention_start_idx = 37
        self.mention_end_idx = 42
        self.mention_loss_ratio = 0.7
        self.is_padding = True
        if self.is_padding:
            self.max_doc_length = 1000
            self.max_candidate_num = int(self.max_doc_length * self.span_ratio)

        # mention proposal 
        self.mention_start_ffnn = nn.Linear(1024, 1)
        self.mention_end_ffnn = nn.Linear(1024, 1)
        self.mention_span_ffnn = nn.Linear(1024 * 2, 1)

        # cluster todo(yuxian): check是否应该和上面的参数不share
        # self.forward_qa_ffnn = nn.Linear(self.config.hidden_size * 2, 1)
        # self.backward_qa_ffnn = nn.Linear(self.config.hidden_size * 2, 1)
        self.mention_link_ffnn = nn.Linear(1024 * 2, 1)
        # self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, sentence_map, subtoken_map, window_input_ids, window_masked_ids, gold_mention_span=None,
                token_type_ids=None,
                attention_mask=None, span_starts=None, span_ends=None, cluster_ids=None, mode="eval"):
        """
        forward
        Args:
            sentence_map: [num_tokens], each token's sentence index
            subtoken_map: [num_tokens], each token's ??? todo
            window_masked_ids: [num_windows, window_size], mask those has appeared in previous window with negative.
            token_type_ids: [num_windows, window_size]
            attention_mask:
            span_starts: [num_spans], span start indices
            span_ends: [num_spans], span end indices
            cluster_ids: [num_spans], span to cluster indices
            gold_mention_span: [num_candidates] todo(yuxian): 和span_start/end重复，没必要传

        Returns:

        """
        num_tokens = sentence_map.shape[0]
        embed_size = 1024
        window_overlap_mask = (window_masked_ids >= 0).long()
        # flattened_input_ids = input_ids.view(-1)
        # flattened_input_mask = input_mask.view(-1)
        # [num_windows, window_size, embed_size] todo(yuxian) attention mask
        mention_sequence_output, _ = self.bert(window_input_ids, token_type_ids, attention_mask,
                                               output_all_encoded_layers=False)
        # [num_tokens]
        doc2windows = self.doc_offsets2window_offsets(window_overlap_mask=window_overlap_mask)
        doc_ids = window_input_ids.view(-1)[doc2windows]

        # [num_candidates], [num_candidates]
        candidate_starts, candidate_ends, candidate_mask = self.get_candidate_spans(sentence_map, doc_ids=doc_ids)
        if self.is_padding:
            k = self.max_candidate_num
        else:
            num_candidate_mentions = max(int(num_tokens * self.span_ratio), 1)
            k = min(self.max_candidate_num, num_candidate_mentions)

        # [num_tokens, embed_size]  todo(yuxian): average overlapped embedding may be better
        doc_embeddings = mention_sequence_output.view(-1, embed_size)[doc2windows]
        # [num_candidates, embed_size]
        candidates_embeddings = self.get_span_embeddings(doc_embeddings,
                                                         candidate_starts,
                                                         candidate_ends)
        # [num_candidates]
        candidate_labels = self.get_candidate_labels(candidate_starts=candidate_starts,
                                                     candidate_ends=candidate_ends,
                                                     labeled_starts=span_starts,
                                                     labeled_ends=span_ends,
                                                     labels=cluster_ids)
        # get topk scores
        # [num_candidates]
        candidate_mention_scores = self.mention_span_ffnn(candidates_embeddings).squeeze(-1)
        # proposal_loss = self.bce_loss(
        #     candidate_mention_scores,
        #     (candidate_labels > 0).float(),
        # )
        # if self.is_padding:
        #     proposal_loss *= candidate_mask.float()
        proposal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            candidate_mention_scores,
            (candidate_labels > 0).float(),
            pos_weight=candidate_mask.float()
        )
        proposal_loss = proposal_loss.mean()
        proposal_loss *= self.mention_loss_ratio
        # [k]  todo(yuxian): 不需要score > 0.0?
        top_mention_scores, top_mention_indices = torch.topk(candidate_mention_scores, k)
        topk_span_starts = candidate_starts[top_mention_indices]
        topk_span_ends = candidate_ends[top_mention_indices]
        topk_span_labels = candidate_labels[top_mention_indices]
        # return (proposal_loss, sentence_map, input_ids, input_mask,
        #         candidate_starts, candidate_ends, candidate_labels, candidate_mention_scores,
        #         topk_span_starts, topk_span_ends, topk_span_labels, top_mention_scores)
        # if mode == "eval":
        #     return topk_span_starts.detach(), topk_span_ends.detach() 
        # else:
        return (proposal_loss, sentence_map.detach(), window_input_ids.detach(), window_masked_ids.detach(),
                candidate_starts.detach(), candidate_ends.detach(), candidate_labels.detach(),
                candidate_mention_scores.detach(),
                topk_span_starts.detach(), topk_span_ends.detach(), topk_span_labels.detach(),
                top_mention_scores.detach())

    @staticmethod
    def doc_offsets2window_offsets(window_overlap_mask):
        """
        map from doc offsets to window offsets
        Args:
            window_overlap_mask: [num_window, window_size] 0 if this token(window_input_ids[i][j])
            is also in previous window(window_input_ids[i-1])

        Returns:
            offsets: [num_tokens], offsets[i] is the first position that doc[i] appear in flattened window input ids.
        """
        return torch.where(window_overlap_mask.view(-1) > 0)[0]

    def batch_qa_linking(self,
        sentence_map,
        window_input_ids,
        window_masked_ids,
        token_type_ids,
        attention_mask,
        candidate_starts,
        candidate_ends,
        candidate_labels,
        candidate_mention_scores,
        topk_span_starts,
        topk_span_ends,
        topk_span_labels,
        topk_mention_scores,
        origin_k,
        gold_mention_span,
        recompute_mention_scores=False,
        mode="train",
    ):
        """

        Args:
            sentence_map: [num_tokens], each token's sentence index
            window_input_ids: [num_windows, window_size]
            window_masked_ids: [num_windows, window_size], mask those has appeared in previous window with negative.
            token_type_ids: [num_windows, window_size]
            attention_mask:
            candidate_starts: [num_candidates]
            candidate_ends: [num_candidates]
            candidate_labels: [num_candidates]
            candidate_mention_scores: [num_candidates]
            topk_span_starts: [k]
            topk_span_ends: [k]
            topk_span_labels: [k]
            topk_mention_scores: [k]
            origin_k: total mention num
            gold_mention_span: [num_candidates] todo(yuxian): 和span_start/end重复，没必要传
            recompute_mention_scores: if True, recompute mention scores
            mode: if "train", only return loss. Return other variables if not train.
        Returns:

        """
        embed_size = self.config.hidden_size
        # [num_tokens]
        doc2windows = self.doc_offsets2window_offsets(window_overlap_mask=(window_masked_ids >= 0).long())
        doc_ids = window_input_ids.view(-1)[doc2windows]
        if recompute_mention_scores:
            # [num_windows, window_size, embed_size]
            mention_sequence_output, _ = self.bert(window_input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
            # [num_tokens, embed_size]  todo(yuxian): average overlapped embedding may be better
            doc_embeddings = mention_sequence_output.view(-1, embed_size)[doc2windows]

            # # [num_candidates]
            # candidates_embeddings = self.get_span_embeddings(doc_embeddings,
            #                                                  candidate_starts,
            #                                                  candidate_ends)
            # get topk scores
            # [k]
            topk_mention_embeddings = self.get_span_embeddings(doc_embeddings,
                                                               topk_span_starts,
                                                               topk_span_ends)
            topk_mention_scores = self.mention_span_ffnn(topk_mention_embeddings).squeeze(-1)

        # flattened_input_ids = window_input_ids.view(-1)
        # flattened_input_mask = window_masked_ids.view(-1)
        k = topk_span_starts.shape[0]
        c = min(self.max_antecedent_num, origin_k)
        # build k queries
        questions = []
        for span_start, span_end in zip(topk_span_starts, topk_span_ends):
            # [query_length]
            # question_tokens = self.get_question_token_ids(sentence_map=sentence_map,
            #                                               flattened_input_ids=flattened_input_ids,
            #                                               flattened_input_mask=flattened_input_mask,
            #                                               span_start=span_start,
            #                                               span_end=span_end)
            question_tokens = self.fast_get_question_token_ids(sentence_map=sentence_map,
                                                               doc_ids=doc_ids,
                                                               span_start=span_start,
                                                               span_end=span_end)
            questions.append(question_tokens)
        # [k, num_windows*window_size, embed_size]
        batch_forward_embeddings = self.get_batch_query_mention_embeddings(
            questions=questions,
            input_ids=window_input_ids
        )
        # [k, num_tokens, embed_size]
        batch_forward_embeddings = batch_forward_embeddings.index_select(dim=1, index=doc2windows)

        # [k, num_candidates, embed_size]
        batch_forward_start_embeddings = torch.index_select(batch_forward_embeddings, dim=1,
                                                            index=candidate_starts)
        batch_forward_end_embeddings = torch.index_select(batch_forward_embeddings, dim=1,
                                                          index=candidate_ends)
        # [k, num_candidates, embed_size*2]
        batch_forward_candidate_embeddings = torch.cat([batch_forward_start_embeddings, batch_forward_end_embeddings],
                                                       dim=-1)
        # [k, num_candidates]  todo(yuxian): maybe add a loss here to make training more stable?
        batch_forward_mention_scores = self.mention_link_ffnn(batch_forward_candidate_embeddings).squeeze(-1)
        # [k, c]
        batch_topc_forward_scores, batch_topc_forward_indices = torch.topk(batch_forward_mention_scores, k=c, dim=1)
        batch_topc_forward_starts = candidate_starts[batch_topc_forward_indices.view(-1)].view(k, c)
        batch_topc_forward_ends = candidate_ends[batch_topc_forward_indices.view(-1)].view(k, c)
        batch_topc_forward_labels = candidate_labels[batch_topc_forward_indices.view(-1)].view(k, c)
        
        if not recompute_mention_scores:
            batch_topc_mention_scores = candidate_mention_scores[batch_topc_forward_indices.view(-1)].view(k, c)
        else:
            batch_topc_mention_embeds = self.get_span_embeddings(doc_embeddings,
                                                                 span_starts=batch_topc_forward_starts.view(-1),
                                                                 span_ends=batch_topc_forward_ends.view(-1))
            batch_topc_mention_scores = self.mention_span_ffnn(batch_topc_mention_embeds).squeeze(-1).view(k, c)

        batch_backward_input_ids = []
        batch_backward_mask = []
        batch_backward_token_type_ids = []
        batch_backward_mention_starts = torch.zeros(k * c, device=window_input_ids.device, dtype=torch.int64)
        batch_backward_mention_ends = torch.zeros(k * c, device=window_input_ids.device, dtype=torch.int64)
        for back_idx, (back_span_start, back_span_end) in enumerate(zip(batch_topc_forward_starts.view(-1),
                                                                        batch_topc_forward_ends.view(-1))):
            k_idx = back_idx // c
            # [query_length]
            # back_question_tokens = self.get_question_token_ids(
            #     sentence_map=sentence_map,
            #     flattened_input_ids=flattened_input_ids,
            #     flattened_input_mask=flattened_input_mask,
            #     span_start=back_span_start,
            #     span_end=back_span_end
            # )
            back_question_tokens = self.fast_get_question_token_ids(
                sentence_map=sentence_map,
                doc_ids=doc_ids,
                span_start=back_span_start,
                span_end=back_span_end
            )
            # use only proposal sent as context
            # [context_length]
            # back_context_tokens, back_mention_start, back_mention_end = self.get_question_token_ids(
            #     sentence_map=sentence_map,
            #     flattened_input_ids=flattened_input_ids,
            #     flattened_input_mask=flattened_input_mask,
            #     span_start=topk_span_starts[k_idx],
            #     span_end=topk_span_ends[k_idx],
            #     return_offset=True
            # )
            back_context_tokens, back_mention_start, back_mention_end = self.fast_get_question_token_ids(
                sentence_map=sentence_map,
                doc_ids=doc_ids,
                span_start=topk_span_starts[k_idx],
                span_end=topk_span_ends[k_idx],
                return_offset=True
            )
            # todo(yuxian): SEP?
            back_input_ids = torch.cat([back_question_tokens, back_context_tokens])
            back_mask = back_input_ids != self.pad_idx
            batch_backward_input_ids.append(back_input_ids)
            batch_backward_mask.append(back_mask)
            batch_backward_token_type_ids.append(torch.cat([torch.zeros_like(back_question_tokens),
                                                            torch.ones_like(back_context_tokens)]))
            # batch_backward_questions.append(back_question_tokens)
            # batch_backward_contexts.append(back_context_tokens)
            batch_backward_mention_starts[back_idx] = back_mention_start + len(back_question_tokens)
            batch_backward_mention_ends[back_idx] = back_mention_end + len(back_context_tokens)
        # [k*c, query_length+context_length]
        batch_backward_input_ids = self.pad_stack(batch_backward_input_ids, pad_value=0, dim=-1)
        batch_backward_mask = self.pad_stack(batch_backward_mask, pad_value=0, dim=-1)
        batch_backward_token_type_ids = self.pad_stack(batch_backward_token_type_ids, pad_value=1, dim=-1)
        # [k*c, query_length+context_length, embed_size]
        batch_backward_embeddings, _ = self.bert(batch_backward_input_ids, batch_backward_token_type_ids,
                                                 batch_backward_mask,
                                                 output_all_encoded_layers=False)
        # [k*c, 1, embed_size]  start_embeddings[i][j] = embeddings[i][starts[i][j][k]][k]
        batch_backward_start_embeddings = batch_backward_embeddings.gather(
            index=batch_backward_mention_starts.unsqueeze(1).unsqueeze(2).expand([-1, -1, self.config.hidden_size]),
            dim=1)
        batch_backward_end_embeddings = batch_backward_embeddings.gather(
            index=batch_backward_mention_ends.unsqueeze(1).unsqueeze(2).expand([-1, -1, self.config.hidden_size]),
            dim=1)
        # [k*c, embed_size*2]
        batch_backward_mention_embeddings = torch.cat([batch_backward_start_embeddings.unsqueeze(1),
                                                       batch_backward_end_embeddings.unsqueeze(1)],
                                                      dim=-1)
        # [k, c]
        batch_backward_mention_scores = self.mention_link_ffnn(batch_backward_mention_embeddings).view(k, c)

        # [k, c]
        cluster_mention_scores = (
            (batch_topc_forward_scores + batch_backward_mention_scores) / 2 +
            topk_mention_scores.unsqueeze(-1) +
            batch_topc_mention_scores
        )

        # (k, c)
        same_cluster_indicator = batch_topc_forward_labels == topk_span_labels.unsqueeze(-1)
        # (k, c)
        pairwise_labels = torch.logical_and(same_cluster_indicator, (topk_span_labels > 0).unsqueeze(-1))
        dummy_labels = torch.logical_not(torch.any(pairwise_labels, dim=1, keepdims=True))  # [k, 1]

        loss_antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1).long()  # [k, c + 1]
        dummy_scores = torch.zeros([k, 1]).to(loss_antecedent_labels.device)
        loss_antecedent_scores = torch.cat([dummy_scores, cluster_mention_scores], 1)  # [k, c + 1]
        link_loss = self.marginal_likelihood(loss_antecedent_scores, loss_antecedent_labels)
        if mode == "train":
            return link_loss
        mention_to_predict = torch.sigmoid(candidate_mention_scores)
        mention_to_predict = mention_to_predict > 0.5
        mention_to_gold = gold_mention_span
        return link_loss, loss_antecedent_scores, mention_to_predict, mention_to_gold

    def marginal_likelihood(self, antecedent_scores, antecedent_labels):
        """
        Desc:
            marginal likelihood of gold antecedent spans from coreference clusters.
        Args:
            antecedent_scores: [k, c+1] the predicted scores by the model
            antecedent_labels: [k, c+1] the gold-truth cluster labels
        Returns:
            a scalar of loss
        """
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float())
        marginalized_gold_scores = torch.logsumexp(gold_scores, 1)  # [k]
        log_norm = torch.logsumexp(antecedent_scores, 1)  # [k]
        loss = log_norm - marginalized_gold_scores  # [k]
        return loss.sum()  # todo(yuxian): sum会不会导致与mention的mean不一致

    def get_candidate_spans(self, sentence_map, doc_ids):
        """
        Desc:
            get candidate spans based on: the length of candidate spans <= max_span_width
            each span is located in a single sentence
        Args:
            sentence_map: [num_tokens], each token's sentence index
            doc_ids: [num_tokens]

        Returns:
            start and end indices w.r.t num_tokens (num_candidates, ), (num_candidates, )
        """
        num_tokens = sentence_map.shape[0]
        # candidate_span: every position can be span start, there are max_span_width kinds of end for each start
        # [num_tokens, max_span_width]
        candidate_starts = torch.arange(0, num_tokens, dtype=torch.int64).unsqueeze(1).expand(
            [-1, self.max_span_width]).contiguous()
        # [num_tokens, max_span_width]
        candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, dtype=torch.int64).unsqueeze(
            0).expand([num_tokens, -1]).contiguous()
        # [num_tokens*max_span_width]
        candidate_starts = candidate_starts.view(-1).to(sentence_map.device)
        candidate_ends = candidate_ends.view(-1).to(sentence_map.device)
        # [num_tokens*max_span_width]，get sentence_id for each token indices
        candidate_start_sentence_indices = sentence_map[candidate_starts]
        candidate_end_sentence_indices = sentence_map[torch.clamp(candidate_ends, min=0, max=num_tokens - 1)]
        # [num_tokens*max_span_width]，legitimate spans should reside in a single sentence.
        candidate_mask = torch.logical_and(
            candidate_ends < num_tokens,
            (candidate_start_sentence_indices - candidate_end_sentence_indices) == 0,
        )
        candidate_mask = torch.logical_and(
            candidate_mask,
            (doc_ids[torch.clamp(candidate_ends, min=0, max=num_tokens - 1)] > 0)
        )
        if self.is_padding:
            candidate_ends = torch.clamp(candidate_ends, min=0, max=num_tokens - 1)
        else:
            candidate_starts = candidate_starts[candidate_mask]
            candidate_ends = candidate_ends[candidate_mask]
        return candidate_starts, candidate_ends, candidate_mask

    def get_span_embeddings(self, token_embeddings, span_starts, span_ends):
        """
        get span embeddings from span start embedding, span end embedding and optionally span width embedding
        Args:
            token_embeddings: [num_tokens, embed_size]
            span_starts: [num_candidates]
            span_ends: [num_candidates]
        Returns:
            span_embeddings: [num_candidates, embed_size*2]
        """
        span_start_emb = token_embeddings[span_starts]  # [num_candidates, embed_size]
        span_end_emb = token_embeddings[span_ends]  # [num_candidates, embed_size]

        span_emb = torch.cat([span_start_emb, span_end_emb], dim=-1)
        return span_emb

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels=None):
        """
            method to get golden cluster id of candidate spans
        Args:
            candidate_starts: [num_candidates, ]
            candidate_ends: [num_candidates, ]
            labeled_starts: [num_mentions, ]
            labeled_ends: [num_mentions, ]
            labels: [num_mentions, ] gold truth cluster ids
        Returns:
            candidate_labels: [num_candidates]
        """
        # [num_mentions, num_candidates]
        same_start = labeled_starts.unsqueeze(1) == candidate_starts.unsqueeze(0)
        # [num_mentions, num_candidates]
        same_end = labeled_ends.unsqueeze(1) == candidate_ends.unsqueeze(0)
        # [num_mentions, num_candidates]
        same_span = torch.logical_and(same_start, same_end)

        # [1, num_candidates]
        candidate_labels = torch.matmul(labels.unsqueeze(0).float(), same_span.float())
        candidate_labels = candidate_labels.long()
        return candidate_labels.squeeze(0)

    def get_question_token_ids(self, sentence_map, flattened_input_ids, flattened_input_mask, span_start, span_end,
                               return_offset=False):
        """
        Desc: # todo(yuxian): add SEP
            construct question based on the selected mention span
        Args:
            sentence_map: [num_tokens] tokens to sentence_id
            flattened_input_ids: [num_windows * window_size]
            flattened_input_mask: [num_windows * window_size] todo(yuxian): this is ugly, consider input origin doc_ids directly
            span_start: int
            span_end: int
            return_offset: bool, if True, return mention start/end
        Returns:
            query tokens: [query_length, ]
        """
        sentence_idx = sentence_map[span_start]
        # [num_sent_token]
        sent_positions = torch.where(sentence_map == sentence_idx)[0]
        sent_start = torch.where(flattened_input_mask == sent_positions[0])[0][0]
        sent_end = torch.where(flattened_input_mask == sent_positions[-1])[0][0]
        origin_tokens = flattened_input_ids[sent_start: sent_end + 1]

        mention_start = torch.where(flattened_input_mask == span_start)[0][0]
        mention_end = torch.where(flattened_input_mask == span_end)[0][0]
        mention_start_in_sentence = mention_start - sent_start
        mention_end_in_sentence = mention_end - sent_start

        question_token_ids = torch.cat([origin_tokens[: mention_start_in_sentence],
                                        torch.Tensor([self.mention_start_idx]).long().to(origin_tokens.device),
                                        origin_tokens[mention_start_in_sentence: mention_end_in_sentence + 1],
                                        torch.Tensor([self.mention_end_idx]).long().to(origin_tokens.device),
                                        origin_tokens[mention_end_in_sentence + 1:],
                                        ], dim=0)
        if return_offset:
            return question_token_ids, mention_start_in_sentence + 1, mention_end_in_sentence + 1
        return question_token_ids

    def fast_get_question_token_ids(self, sentence_map, doc_ids, span_start, span_end,
                                    return_offset=False):
        """
        Desc: # todo(yuxian): add SEP
            construct question based on the selected mention span
        Args:
            sentence_map: [num_tokens] tokens to sentence_id
            doc_ids: [num_tokens] doc tokens
            span_start: int
            span_end: int
            return_offset: bool, if True, return mention start/end
        Returns:
            query tokens: [query_length] query_length=origin_sent_length+2(special token)
        """
        sentence_idx = sentence_map[span_start]
        # [num_sent_token]
        sent_positions = torch.where(sentence_map == sentence_idx)[0]
        sent_start = sent_positions[0]
        sent_end = sent_positions[-1]
        origin_tokens = doc_ids[sent_start: sent_end + 1]

        mention_start_in_sentence = span_start - sent_start
        mention_end_in_sentence = span_end - sent_start

        question_token_ids = torch.cat([origin_tokens[: mention_start_in_sentence],
                                        torch.Tensor([self.mention_start_idx]).long().to(origin_tokens.device),
                                        origin_tokens[mention_start_in_sentence: mention_end_in_sentence + 1],
                                        torch.Tensor([self.mention_end_idx]).long().to(origin_tokens.device),
                                        origin_tokens[mention_end_in_sentence + 1:],
                                        ], dim=0)
        if return_offset:
            return question_token_ids, mention_start_in_sentence + 1, mention_end_in_sentence + 1
        return question_token_ids

    def get_query_mention_embeddings(self, question_tokens, input_ids):
        """
        get embedding of all candidates with respect to question
        Args:
            question_tokens: [query_length]
            input_ids: [num_windows, window_size]

        Returns:
            query mention embeddings, [num_window*window_size, hidden]

        """
        num_windows = input_ids.shape[0]
        query_length = question_tokens.shape[0]
        # [num_windows, query_length]
        windows_questions = question_tokens.unsqueeze(0).expand([num_windows, -1])
        question_ones = torch.ones_like(windows_questions)
        question_zeros = torch.zeros_like(windows_questions)
        actual_mask = (input_ids == self.pad_idx).long()  # (num_windows, window_size)
        # [num_windows, query_length + window_size]
        qa_input_ids = torch.cat([windows_questions, input_ids], 1)
        # [num_windows, query_length + window_size]
        qa_input_mask = torch.cat([question_ones, actual_mask], 1)
        token_type_ids = torch.cat([question_zeros, torch.ones_like(input_ids)], 1)
        # [num_windows, query_length + window_size, embed_size]
        forward_embeddings, _ = self.bert(input_ids=qa_input_ids, attention_mask=qa_input_mask,
                                          token_type_ids=token_type_ids, output_all_encoded_layers=False)
        # [num_windows, window_size, embed_size]
        forward_embeddings = forward_embeddings[:, query_length:]
        # [num_windows*window_size, embed_size]
        flattened_forward_embeddings = forward_embeddings.view(-1, self.config.hidden_size)
        return flattened_forward_embeddings

    def get_batch_query_mention_embeddings(self, questions, input_ids):
        """
        get embedding of all candidates with respect to question. concatenate all questions as batch
        Args:
            questions: list of different sizes tensor of shape [query_length]
            input_ids: [num_windows, window_size]
        Returns:
            flattened_embeddings: [num_questions, num_windws*window_size, embed_size]
        """
        num_questions = len(questions)
        num_windows, window_size = input_ids.shape
        batch_input_ids = []
        batch_input_mask = []
        batch_token_type_ids = []
        for question_tokens in questions:
            # [num_windows, query_length]
            windows_questions = question_tokens.unsqueeze(0).expand([num_windows, -1])
            question_ones = torch.ones_like(windows_questions)
            question_zeros = torch.zeros_like(windows_questions)
            actual_mask = (input_ids == self.pad_idx).long()  # (num_windows, window_size)
            # [num_windows, query_length + window_size]
            qa_input_ids = torch.cat([windows_questions, input_ids], 1)
            # [num_windows, query_length + window_size]
            qa_input_mask = torch.cat([question_ones, actual_mask], 1)
            # todo(yuxian): ones or zeros?
            token_type_ids = torch.cat([question_zeros, torch.ones_like(input_ids)], 1)
            batch_input_ids.append(qa_input_ids)
            batch_input_mask.append(qa_input_mask)
            batch_token_type_ids.append(token_type_ids)
        # [batch*num_windows, max_seq_len]
        batch_input_ids = self.pad_stack(batch_input_ids).view(num_questions * num_windows, -1)
        batch_input_mask = self.pad_stack(batch_input_mask).view(num_questions * num_windows, -1)
        # todo(yuxian): 1 or 0 ?
        batch_token_type_ids = self.pad_stack(batch_token_type_ids, 1).view(num_questions * num_windows, -1)

        # [batch*num_windows, max_seq_len, embed_size]
        forward_embeddings, _ = self.bert(input_ids=batch_input_ids, attention_mask=batch_input_mask,
                                          token_type_ids=batch_token_type_ids, output_all_encoded_layers=False)
        # [batch, num_windows, max_seq_len, embed_size]
        forward_embeddings = forward_embeddings.view(num_questions, num_windows, -1, self.config.hidden_size)
        # [batch, num_windows, window_size, embed_size]
        truncated_embeddings = [
            embeddings[:, question.shape[0]: question.shape[0] + window_size:, ]
            for embeddings, question in zip(forward_embeddings, questions)
        ]
        truncated_embeddings = torch.stack(truncated_embeddings, 0)
        flattened_forward_embeddings = truncated_embeddings.view(num_questions, -1, self.config.hidden_size)
        return flattened_forward_embeddings

    @staticmethod
    def pad_stack(tensors, pad_value: int = 0, dim: int = -1):
        """
        stack tensors and pad to same max length
        note that all tensors must have same length in all dimensions except dim.
        Args:
            tensors: List of tensor, has different length at dim
            pad_value: pad value
            dim: pad dimension.
        Returns:
            batch_tensor: [batch, tensor.shape]
        """
        max_length = max(t.shape[dim] for t in tensors)
        first_tensor_shape = list(tensors[0].shape)
        final_shape = first_tensor_shape
        final_shape[dim] = max_length
        tensor_num = len(tensors)
        output_tensor = torch.zeros([tensor_num] + final_shape, dtype=tensors[0].dtype, device=tensors[0].device)
        output_tensor.fill_(pad_value)
        for tensor_idx, tensor in enumerate(tensors):
            slices = [slice(None, size, None) for size in tensor.shape]
            output_tensor[tensor_idx][slices] = tensor
        return output_tensor
