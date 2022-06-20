from dataLoader import Corpus
import torch
import pytorch_lightning as pl
from itertools import chain
import torch.nn as nn
from transformers import AdamW
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import networkx as nx
from config import arg_parse
from utils import extract_gold_corefs, safe_divide, extract_gold_coref_cluster, extract_pred_coref_cluster, muc, b_cubed, ceaf_phi3, ceaf_phi4, conll_coref_f1
from Networks import CorefQA
from dataLoader import Corpus
from subprocess import Popen, PIPE

import numpy as np

args = arg_parse()

class corefQAModel(pl.LightningModule):

    def __init__(self, segment_max_num=args.segment_max_num,
                       sentense_max_num=args.sentense_max_num):
        super().__init__()

        # -------------Model-Related--------------------------------
        self.CorefScorer = CorefQA()

        # -------------Train-Related-------------------------------
        if args.bert_based:
            self.parts_max_num = segment_max_num
        else:
            self.parts_max_num = sentense_max_num
    
    def forward(self, doc):
        """ Encode document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores..
        """
        
        proposal_loss, sentence_map, window_input_ids, window_masked_ids, candidate_starts, candidate_ends, candidate_labels, candidate_mention_scores, topk_span_starts, topk_span_ends, topk_span_labels, topk_mention_scores = self.CorefScorer.forward(
                sentence_map=doc.sentence_map.squeeze(0),
                subtoken_map=doc.subtoken_map,
                window_input_ids=doc.flattened_window_input_ids.view(-1, 128),
                window_masked_ids=doc.flattened_window_masked_ids.view(-1, 128),
                gold_mention_span=doc.mention_span.squeeze(0),
                token_type_ids=None,
                attention_mask=None,
                span_starts=doc.span_starts.squeeze(0),
                span_ends=doc.span_ends.squeeze(0),
                cluster_ids=doc.cluster_ids.squeeze(0)
            )

        mention_chunk_size = 1
        mention_num = topk_span_starts.shape[0]
        for chunk_idx in range(mention_num):
                    chunk_start = mention_chunk_size * chunk_idx
                    chunk_end = chunk_start + mention_chunk_size
                    link_loss = self.CorefScorer.batch_qa_linking(
                        sentence_map=sentence_map.squeeze(0),
                        window_input_ids=window_input_ids,
                        window_masked_ids=window_masked_ids,
                        token_type_ids=None,
                        attention_mask=None,
                        candidate_starts=candidate_starts,
                        candidate_ends=candidate_ends,
                        candidate_labels=candidate_labels,
                        candidate_mention_scores=candidate_mention_scores,
                        topk_span_starts=topk_span_starts[chunk_start: chunk_end],
                        topk_span_ends=topk_span_ends[chunk_start: chunk_end],
                        topk_span_labels=topk_span_labels[chunk_start: chunk_end],
                        topk_mention_scores=topk_mention_scores[chunk_start: chunk_end],
                        origin_k=mention_num,
                        gold_mention_span=doc.mention_span.squeeze(0),
                        recompute_mention_scores=True
                    )

        return proposal_loss, link_loss

    def shared_step(self, batch, stage):
        
        document = batch

        # Randomly s document to up to 50 sentences/2 segments
        # document = document.truncate()

        # Extract gold coreference links
        gold_corefs, total_corefs, \
            gold_mentions, total_mentions = extract_gold_corefs(document)
        
        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Init metrics
        correct_mentions, corefs_found, corefs_chosen = 0, 0, 0
        
        # Predict coref probabilites for each span in a document
        spans, scores = self.forward(document)
        
        # If too few spans were found
        if spans is None:
            return (0, 0, 0, 0, 0, 0)
        
        # Get log-likelihood of correct antecedents implied by gold clustering
        gold_indexes = torch.zeros_like(scores).to(self.device)
        predict_mentions = len(spans)
        for idx, span in enumerate(spans):
            
            # Log number of mentions found
            if (span.i1, span.i2) in gold_mentions:
                correct_mentions += 1
                # Check which of these tuples are in the gold set, if any
                golds = [
                    i+1 for i, link in enumerate(span.yi_idx)
                    if link in gold_corefs
                ]

                # If gold_pred_idx is not empty, consider the probabilities of the found antecedents
                if golds:
                    gold_indexes[idx, golds] = 1

                    # Progress logging for recall
                    corefs_found += len(golds)
                    found_corefs = sum((scores[idx, golds] > scores[idx, 0])).detach()
                    corefs_chosen += found_corefs.item()
                else:
                    # Otherwise, set gold to dummy
                    gold_indexes[idx, 0] = 1

            else:
                # Otherwise, set gold to dummy
                gold_indexes[idx, 0] = 1

        loss = self.softmax_loss(scores, gold_indexes)

        gold_coref_cluster = extract_gold_coref_cluster(document)
        pred_coref_cluster = extract_pred_coref_cluster(spans, scores)

        muc_precision, muc_recall, muc_f1 = muc(pred_coref_cluster, gold_coref_cluster)
        bcubed_precision, bcubed_recall, bcubed_f1 = b_cubed(pred_coref_cluster, gold_coref_cluster)
        ceaf_3_precision, ceaf_3_recall, ceaf_3_f1 = ceaf_phi3(pred_coref_cluster, gold_coref_cluster)
        ceaf_4_precision, ceaf_4_recall, ceaf_4_f1 = ceaf_phi4(pred_coref_cluster, gold_coref_cluster)
        avg_f1 = (muc_f1 + bcubed_f1 + ceaf_3_f1) / 3 

        return {
            'loss': loss,
            'correct_mentions': correct_mentions,
            'gold_mentions': total_mentions, # len(gold_mentions)
            'predict_mentions': predict_mentions,
            'corefs_found': corefs_found,
            'corefs_chosen': corefs_chosen,
            'total_corefs': total_corefs,
            'muc_precision': muc_precision,
            'muc_recall': muc_recall, 
            'muc_f1': muc_f1,
            'bcubed_precision': bcubed_precision,
            'bcubed_recall': bcubed_recall,
            'bcubed_f1': bcubed_f1,
            'ceaf_3_precision': ceaf_3_precision,
            'ceaf_3_recall': ceaf_3_recall,
            'ceaf_3_f1': ceaf_3_f1,
            'ceaf_4_precision': ceaf_4_precision,
            'ceaf_4_recall': ceaf_4_recall,
            'ceaf_4_f1': ceaf_4_f1,
            'avg_f1': avg_f1
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        correct_mentions = sum([x["correct_mentions"] for x in outputs])
        gold_mentions = sum([x["gold_mentions"] for x in outputs])
        predict_mentions = sum([x["predict_mentions"] for x in outputs])
        corefs_found = sum([x["corefs_found"] for x in outputs])
        corefs_chosen = sum([x["corefs_chosen"] for x in outputs])
        total_corefs = sum([x["total_corefs"] for x in outputs])

        epoch_muc_precision = np.mean([x["muc_precision"] for x in outputs])
        epoch_muc_recall = np.mean([x["muc_recall"] for x in outputs])
        epoch_muc_f1 = np.mean([x["muc_f1"] for x in outputs])

        epoch_bcubed_precision = np.mean([x["bcubed_precision"] for x in outputs])
        epoch_bcubed_recall = np.mean([x["bcubed_recall"] for x in outputs])
        epoch_bcubed_f1 = np.mean([x["bcubed_f1"] for x in outputs])

        epoch_ceaf_3_precision = np.mean([x["ceaf_3_precision"] for x in outputs])
        epoch_ceaf_3_recall = np.mean([x["ceaf_3_recall"] for x in outputs])
        epoch_ceaf_3_f1 = np.mean([x["ceaf_3_f1"] for x in outputs])

        epoch_ceaf_4_precision = np.mean([x["ceaf_4_precision"] for x in outputs])
        epoch_ceaf_4_recall = np.mean([x["ceaf_4_recall"] for x in outputs])
        epoch_ceaf_4_f1 = np.mean([x["ceaf_4_f1"] for x in outputs])

        epoch_avg_f1 = np.mean([x["avg_f1"] for x in outputs])

        metrics = {
            f"{stage}_mentions_recall": safe_divide(correct_mentions, gold_mentions),
            f"{stage}_mentions_precision": safe_divide(correct_mentions, predict_mentions),
            f"{stage}_coref_recall": safe_divide(corefs_found, total_corefs),
            f"{stage}_coref_precision": safe_divide(corefs_chosen, total_corefs),
            f"{stage}_muc_precision": epoch_muc_precision,
            f"{stage}_muc_recall": epoch_muc_recall,
            f"{stage}_muc_f1": epoch_muc_f1,
            f"{stage}_bcubed_precision": epoch_bcubed_precision,
            f"{stage}_bcubed_recall": epoch_bcubed_recall,
            f"{stage}_bcubed_f1": epoch_bcubed_f1,
            f"{stage}_ceaf_3_precision": epoch_ceaf_3_precision,
            f"{stage}_ceaf_3_recall": epoch_ceaf_3_recall,
            f"{stage}_ceaf_3_f1": epoch_ceaf_3_f1,
            f"{stage}_ceaf_4_precision": epoch_ceaf_4_precision,
            f"{stage}_ceaf_4_recall": epoch_ceaf_4_recall,
            f"{stage}_ceaf_4_f1": epoch_ceaf_4_f1,
            f"{stage}_avg_f1": epoch_avg_f1,
        }

        self.log_dict(metrics, prog_bar=True)
        self.logger.log_metrics(metrics)
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def softmax_loss(self, scores, gold_indexes):
        gold_scores = scores + torch.log(gold_indexes)
        marginalized_gold_scores = torch.logsumexp(gold_scores, 1) 
        log_norm = torch.logsumexp(scores, 1) 
        loss = log_norm - marginalized_gold_scores
        
        return torch.sum(loss)
    
    def configure_optimizers(self):
        param_optimizer = list(self.CorefScorer.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=10e-8)

        return self.optimizer