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
from utils import extract_gold_corefs, safe_divide, extract_gold_coref_cluster, extract_pred_coref_cluster, muc, b_cubed, ceaf_phi3, ceaf_phi4, lea, conll_coref_f1
from Networks import BertCorefScorer
from dataLoader import Corpus
from subprocess import Popen, PIPE

import numpy as np

args = arg_parse()

class bertCorefModel(pl.LightningModule):

    def __init__(self, distribute_model=args.distribute_model,
                       attn_dim = args.atten_dim,
                       embeds_dim = args.embeds_dim,
                       distance_dim=args.distance_dim,
                       genre_dim=args.genre_dim,
                       speaker_dim=args.speaker_dim,
                       segment_max_num=args.segment_max_num,
                       sentense_max_num=args.sentense_max_num):
        super().__init__()

        # -------------Model-Related--------------------------------
        self.CorefScorer = BertCorefScorer()

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

        spans, coref_scores = self.CorefScorer.forward(doc)

        return spans, coref_scores

    def shared_step(self, batch, stage):
        
        document = batch

        # Randomly s document to up to 50 sentences/2 segments
        document = document.truncate()

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
        lea_precision, lea_recall, lea_f1 = lea(pred_coref_cluster, gold_coref_cluster)
        avg_f1 = (muc_f1 + bcubed_f1 + ceaf_3_f1) / 3 

        return {
            'loss': loss,
            'tracked_loss': loss.item(),
            'correct_mentions': correct_mentions,
            'gold_mentions': total_mentions, # len(gold_mentions)
            'predict_mentions': predict_mentions,
            # 'corefs_found': corefs_found,
            # 'corefs_chosen': corefs_chosen,
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
            'lea_precision': lea_precision,
            'lea_recall': lea_recall,
            'lea_f1': lea_f1,
            'avg_f1': avg_f1
        }

    def shared_epoch_end(self, outputs, stage):
        epoch_avg_loss = np.mean([x["tracked_loss"] for x in outputs])
        # aggregate step metics
        correct_mentions = sum([x["correct_mentions"] for x in outputs])
        gold_mentions = sum([x["gold_mentions"] for x in outputs])
        predict_mentions = sum([x["predict_mentions"] for x in outputs])
        # corefs_found = sum([x["corefs_found"] for x in outputs])
        # corefs_chosen = sum([x["corefs_chosen"] for x in outputs])
        # total_corefs = sum([x["total_corefs"] for x in outputs])

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
        
        epoch_lea_precision = np.mean([x['lea_precision'] for x in outputs])
        epoch_lea_recall = np.mean([x['lea_recall'] for x in outputs])
        epoch_lea_f1= np.mean([x['lea_f1'] for x in outputs])

        epoch_avg_f1 = np.mean([x["avg_f1"] for x in outputs])

        metrics = {
            f"{stage}_loss": epoch_avg_loss,
            f"{stage}_mentions_recall": safe_divide(correct_mentions, gold_mentions),
            f"{stage}_mentions_recall": safe_divide(correct_mentions, gold_mentions),
            f"{stage}_mentions_precision": safe_divide(correct_mentions, predict_mentions),
            # f"{stage}_coref_recall": safe_divide(corefs_found, total_corefs),
            # f"{stage}_coref_precision": safe_divide(corefs_chosen, total_corefs),
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
            f"{stage}_lea_precision": epoch_lea_precision,
            f"{stage}_lea_recall": epoch_lea_recall,
            f"{stage}_lea_f1": epoch_lea_f1,
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

        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer = AdamW(params=[
                                        {'params': [p for n, p in chain(self.CorefScorer.score_spans.named_parameters(), self.CorefScorer.score_pairs.named_parameters()) if any(nd in n for nd in no_decay)],'weight_decay': 0.0},

                                        {'params': [p for n, p in chain(self.CorefScorer.score_spans.named_parameters(),self.CorefScorer.score_pairs.named_parameters())if not any(nd in n for nd in no_decay)]},

                                        {'params': [p for n, p in self.CorefScorer.encoder.named_parameters() 
                                                    if any(nd in n for nd in no_decay)], 
                                                     'weight_decay': 0.0, 'lr': args.bert_lr},

                                        {'params': [p for n, p in self.CorefScorer.encoder.named_parameters() 
                                                    if not any(nd in n for nd in no_decay)], 
                                                     'lr': args.bert_lr}
                                    ],
                                    lr=args.lr, weight_decay=0.01)
        
        if args.scheduler == 'ExponentialLR':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.scheduler_gamma, last_epoch=-1, verbose=False)
        elif args.scheduler == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.scheduler_T_max, eta_min=0, last_epoch=-1)

        if args.scheduler == 'None':
            return self.optimizer
        else:
            return [self.optimizer], [self.scheduler]