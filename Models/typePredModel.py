import sys 
sys.path.append("..") 
import torch
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AdamW
import torch.optim as optim

from config import arg_parse

from Networks import typePredictor

import json

from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import classification_report

import numpy as np

args = arg_parse()

class typePredModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # -------------Model-Related--------------------------------
        self.typePredictor = typePredictor()

        self.loss = nn.CrossEntropyLoss()

    
    def forward(self, batched_sents, spans):

        pred_types = self.typePredictor.forward(batched_sents, spans)

        return pred_types

    def shared_step(self, batch, stage):
        
        batched_sents, spans, ture_types = batch

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Predict coref probabilites for each span in a document
        pred_types = self.forward(batched_sents, spans)
        ture_types = torch.tensor(ture_types, dtype=torch.long).to(args.device)
        loss = self.loss(pred_types, ture_types)

        # for criterion
        pred_types = torch.argmax(nn.functional.softmax(pred_types, dim=1),  dim=1).tolist()
        ture_types = ture_types.tolist()

        # with open('./Dataset/others/type2label.json', 'r') as fs:
        #     target_names = json.load(fs).keys()
        # report = classification_report(ture_types, pred_types, target_names=target_names)

        weighted_precision = precision_score(ture_types, pred_types, average='weighted', zero_division=0)
        weighted_recall = recall_score(ture_types, pred_types, average='weighted', zero_division=0)
        weighted_f1 = f1_score(ture_types, pred_types, average='weighted', zero_division=0)

        micro_precision = precision_score(ture_types, pred_types, average='micro', zero_division=0)
        micro_recall = recall_score(ture_types, pred_types, average='micro', zero_division=0)
        micro_f1 = f1_score(ture_types, pred_types, average='micro', zero_division=0)

        accuracy = accuracy_score(ture_types, pred_types)


        return {
            'loss': loss,
            'tracked_loss': loss.item(),
            'weighted_precision':weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'accuracy':accuracy
        }

    def shared_epoch_end(self, outputs, stage):
        epoch_avg_loss = np.mean([x["tracked_loss"] for x in outputs])

        # aggregate step metics
        epoch_weighted_precision = np.mean([x["weighted_precision"] for x in outputs])
        epoch_weighted_recall = np.mean([x["weighted_recall"] for x in outputs])
        epoch_weighted_f1 = np.mean([x["weighted_f1"] for x in outputs])

        epoch_micro_precision = np.mean([x["micro_precision"] for x in outputs])
        epoch_micro_recall = np.mean([x["micro_recall"] for x in outputs])
        epoch_micro_f1 = np.mean([x["micro_f1"] for x in outputs])

        epoch_accuracy = np.mean([x["accuracy"] for x in outputs])

        metrics = {
            f"{stage}_loss": epoch_avg_loss,
            f"{stage}_weighted_precision": epoch_weighted_precision,
            f"{stage}_weighted_recall": epoch_weighted_recall,
            f"{stage}_weighted_f1": epoch_weighted_f1,
            f"{stage}_micro_precision": epoch_micro_precision,
            f"{stage}_micro_recall": epoch_micro_recall,
            f"{stage}_micro_f1": epoch_micro_f1,
            f"{stage}_accurary": epoch_accuracy,
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
                                        {'params': [p for n, p in self.typePredictor.scorer.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0},

                                        {'params': [p for n, p in self.typePredictor.scorer.named_parameters()if not any(nd in n for nd in no_decay)]},

                                        {'params': [p for n, p in self.typePredictor.encoder.named_parameters() 
                                                    if any(nd in n for nd in no_decay)], 
                                                     'weight_decay': 0.0, 'lr': args.bert_lr},

                                        {'params': [p for n, p in self.typePredictor.encoder.named_parameters() 
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