from dataLoader import Corpus
import torch
import pytorch_lightning as pl
from itertools import chain
import torch.nn as nn
from transformers import AdamW
import transformers
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import networkx as nx
from config import arg_parse
from utils import extract_gold_corefs, safe_divide, extract_gold_coref_cluster, extract_pred_coref_cluster, muc, b_cubed, ceaf_phi3, ceaf_phi4, lea, conll_coref_f1
from Networks import wordLevelCorefScorer
from dataLoader import Corpus
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

args = arg_parse()

class wordLevelModel(pl.LightningModule):

    def __init__(self, train_corpus_len=args.corpus_len,
                       segment_max_num=args.segment_max_num,
                       sentense_max_num=args.sentense_max_num):
        super().__init__()

        # -------------Model-Related--------------------------------
        self.CorefScorer = wordLevelCorefScorer()

        # -------------Train-Related-------------------------------
        self._coref_criterion = CorefLoss(args.wl_bce_loss_weight)
        self._span_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if args.bert_based:
            self.parts_max_num = segment_max_num
        else:
            self.parts_max_num = sentense_max_num
        
        self.train_corpus_len = train_corpus_len
    
    def forward(self, doc):
        """ Encode document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores..
        """

        result = self.CorefScorer.forward(doc)

        return result

    def shared_step(self, batch, batch_idx, stage):
        
        document = batch

        # Randomly s document to up to 50 sentences/2 segments
        # document = document.truncate()

        # Extract gold coreference links
        gold_corefs, total_corefs, \
            gold_mentions, gold_mentions_len = extract_gold_corefs(document)
        
        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Init metrics
        correct_mentions, mentions_found, corefs_found, corefs_chosen = 0, 0, 0, 0
        
        # Predict coref probabilites for each span in a document
        result = self.forward(document)

        c_loss = self._coref_criterion(result.coref_scores, result.coref_y)
        if result.span_y:
            s_loss = (self._span_criterion(result.span_scores[:, :, 0], result.span_y[0])
                               + self._span_criterion(result.span_scores[:, :, 1], result.span_y[1])) / len(document.head2span) / 2

        else:
            s_loss = torch.zeros_like(c_loss)

        if result.span_y:
            # pred_starts = result.span_scores[:, :, 0].argmax(dim=1)
            # pred_ends = result.span_scores[:, :, 1].argmax(dim=1)
            # correct_mentions += ((result.span_y[0] == pred_starts) * (result.span_y[1] == pred_ends)).sum().item()
            # mentions_found += len(pred_starts)

            mentions_found = len([span for cluster in result.span_clusters for span in cluster])
            correct_mentions = len([span for cluster in result.span_clusters for span in cluster if span in gold_mentions])


        # If too few spans were found
        # if mentions_found == 0:
        #     return (0, 0, 0, 0, 0, 0)
        
        gold_coref_cluster = document.span_clusters
        pred_coref_cluster = result.span_clusters

        muc_precision, muc_recall, muc_f1 = muc(pred_coref_cluster, gold_coref_cluster)
        bcubed_precision, bcubed_recall, bcubed_f1 = b_cubed(pred_coref_cluster, gold_coref_cluster)
        ceaf_3_precision, ceaf_3_recall, ceaf_3_f1 = ceaf_phi3(pred_coref_cluster, gold_coref_cluster)
        ceaf_4_precision, ceaf_4_recall, ceaf_4_f1 = ceaf_phi4(pred_coref_cluster, gold_coref_cluster)
        lea_precision, lea_recall, lea_f1 = lea(pred_coref_cluster, gold_coref_cluster)
        avg_f1 = (muc_f1 + bcubed_f1 + ceaf_3_f1) / 3 

        return {
            'loss': c_loss+s_loss,
            'coref_loss': c_loss.item(),
            'span_loss': s_loss.item(),
            'correct_mentions': correct_mentions,
            'gold_mentions': len(gold_mentions), # len(gold_mentions)
            'predict_mentions': mentions_found,
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
        
        epoch_avg_loss = np.mean([x["loss"].item() for x in outputs])
        epoch_avg_coref_loss = np.mean([x["coref_loss"] for x in outputs])
        epoch_avg_span_loss =  np.mean([x["span_loss"] for x in outputs])

        metrics = {
            f"{stage}_loss": epoch_avg_loss,
            f"{stage}_avg_coref_loss": epoch_avg_coref_loss,
            f"{stage}_avg_span_loss": epoch_avg_span_loss,   
            f"{stage}_mentions_recall": safe_divide(correct_mentions, gold_mentions),
            f"{stage}_mentions_precision": safe_divide(correct_mentions, predict_mentions),         
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
        return self.shared_step(batch, batch_idx, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "valid")

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
    
    #--------------Interesting Expriments with lr and scheduler----------------
    def configure_optimizers(self):
        for param in self.CorefScorer.bert.parameters():
            param.requires_grad = not args.freeze_bert
        
        modules = sorted((key, value) for key, value in self.CorefScorer.trainable.items() if key != "bert")
        params = []
        for _, module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)
        
        self.optimizer = torch.optim.Adam(
            [{'params': self.CorefScorer.bert.parameters(), 'lr': args.bert_lr},
            {'params': params, 'lr': args.lr}]
        )

        if args.scheduler == 'ExponentialLR':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.scheduler_gamma, last_epoch=-1, verbose=False)
        elif args.scheduler == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.scheduler_T_max, eta_min=0, last_epoch=-1)

        
        if args.scheduler == 'None':
            return self.optimizer
        else:
            return [self.optimizer], [self.scheduler]
    
    # def configure_optimizers(self):
    def configure_optimizers_onelr(self):
     
        modules = sorted((key, value) for key, value in self.CorefScorer.trainable.items())
        params = []
        for _, module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)
        
        self.optimizer = torch.optim.Adam(params, lr=args.lr)
        # self.optimizer = torch.optim.Adam(params, lr=args.bert_lr)
 
        return self.optimizer
    
    # def configure_optimizers(self):
    def configure_optimizers_withscheduler(self):
        for param in self.CorefScorer.bert.parameters():
            param.requires_grad = not args.freeze_bert
        
        modules = sorted((key, value) for key, value in self.CorefScorer.trainable.items() if key != "bert")
        params = []
        for _, module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)
        
        self.optimizer = torch.optim.Adam(
            [{'params': self.CorefScorer.bert.parameters(), 'lr': args.bert_lr},
            {'params': params, 'lr': args.lr}]
        )
        
        self.scheduler = transformers.get_linear_schedule_with_warmup(
                    self.optimizer,
                    self.train_corpus_len, self.train_corpus_len * args.max_epochs
                )

        return [self.optimizer], [self.scheduler]
    
    def configure_optimizers_onelr_withscheduler(self):
        modules = sorted((key, value) for key, value in self.CorefScorer.trainable.items())
        params = []
        for _, module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)
        
        self.optimizer = torch.optim.Adam(params, lr=args.lr)
        # self.optimizer = torch.optim.Adam(params, lr=args.bert_lr)
 
        self.scheduler = transformers.get_linear_schedule_with_warmup(
                    self.optimizer,
                    self.train_corpus_len, self.train_corpus_len * args.max_epochs
                )

        return self.optimizer, self.scheduler


class CorefLoss(torch.nn.Module):
    """ See the rationale for using NLML in Lee et al. 2017
    https://www.aclweb.org/anthology/D17-1018/
    The added weighted summand of BCE helps the model learn even after
    converging on the NLML task. """

    def __init__(self, bce_weight: float):
        assert 0 <= bce_weight <= 1
        super().__init__()
        self._bce_module = torch.nn.BCEWithLogitsLoss()
        self._bce_weight = bce_weight

    def forward(self,    # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                input_: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """ Returns a weighted sum of two losses as a torch.Tensor """
        return (self._nlml(input_, target)
                + self._bce(input_, target) * self._bce_weight)

    def _bce(self,
             input_: torch.Tensor,
             target: torch.Tensor) -> torch.Tensor:
        """ For numerical stability, clamps the input before passing it to BCE.
        """
        return self._bce_module(torch.clamp(input_, min=-50, max=50), target)

    @staticmethod
    def _nlml(input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gold = torch.logsumexp(input_ + torch.log(target), dim=1)
        input_ = torch.logsumexp(input_, dim=1)
        return (input_ - gold).mean() 