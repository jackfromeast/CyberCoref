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
from Networks import CyberCorefScorer, SentCorelModel
from dataLoader import Corpus
from subprocess import Popen, PIPE

import numpy as np

args = arg_parse()

class cyberCorefModel(pl.LightningModule):

    def __init__(self, MaxSentLen,
                       segment_max_num=args.segment_max_num,
                       sentense_max_num=args.sentense_max_num):
        super().__init__()

        # -------------Model-Related--------------------------------
        self.SentCorelModel = SentCorelModel(MaxSentLen)

        # -------------Train-Related-------------------------------
        if args.bert_based:
            self.parts_max_num = segment_max_num
        else:
            self.parts_max_num = sentense_max_num
        
        self.loss = torch.nn.BCELoss(reduction="sum")
    
    def forward(self, doc):
        """ Encode document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores..
        """

        scores, sentpair_index = self.SentCorelModel.forward(doc)

        return scores, sentpair_index
    
    def shared_step(self, batch, stage):
        
        document = batch

        # Randomly s document to up to 50 sentences/2 segments
        document = document.truncate()

        # Extract gold coreference links
        gold_corefs, total_corefs, \
            gold_mentions, total_mentions = extract_gold_corefs(document)
        
        # Zero out optimizer gradients
        self.optimizer.zero_grad()
        
        # Predict coref probabilites for each span in a document
        scores, sentpair_index= self.forward(document)

        # Get sentpair label

        # Compute Loss

        # Compute Accuary, Precision, Recall, F1

        

        return {
            'loss': loss,
            'tracked_loss': loss.item(),
            'accuary': accuary,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def shared_epoch_end(self, outputs, stage):
        epoch_avg_loss = np.mean([x["tracked_loss"] for x in outputs])

        epoch_accuary = np.mean([x["accuary"] for x in outputs])
        epoch_precision = np.mean([x["precision"] for x in outputs])
        epoch_recall = np.mean([x["recall"] for x in outputs])
        epoch_f1 = np.mean([x["f1"] for x in outputs])

        metrics = {
            f"{stage}_loss": epoch_avg_loss,
            f"{stage}_accuary": epoch_accuary,
            f"{stage}_precision": epoch_precision,
            f"{stage}_recall": epoch_recall,
            f"{stage}_f1": epoch_f1,
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

    def configure_optimizers(self):

        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer = AdamW(params=[
                                        {'params': [p for n, p in chain(self.SentCorelModel.SentCorelationAnalyzer.named_parameters(), self.CorefScorer.SentCorelScore.named_parameters()) if any(nd in n for nd in no_decay)],'weight_decay': 0.0},

                                        {'params': [p for n, p in chain(self.SentCorelModel.SentCorelationAnalyzer.named_parameters(),self.CorefScorer.SentCorelScore.named_parameters())if not any(nd in n for nd in no_decay)]},

                                        {'params': [p for n, p in self.SentCorelModel.encoder.named_parameters() 
                                                    if any(nd in n for nd in no_decay)], 
                                                     'weight_decay': 0.0, 'lr': args.bert_lr},

                                        {'params': [p for n, p in self.SentCorelModel.encoder.named_parameters() 
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