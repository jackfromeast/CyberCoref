import os
from config import arg_parse
from dataLoader import load_corpus, Corpus, Document, BERTDocument, corefQADocument, wordLevelDocument, CyberDocument
from Models import bertCorefModel, wordLevelModel, cyberCorefModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from torch.utils.data import DataLoader

"""
Training examples:
+ For End-to-End Neural Networks Model:
    python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_nn.pkl --max_epochs 50 --model nnCorefModel --hidden_dim 200 --embeds_dim 400 --distance_dim 32 --max_span_length 20 --prune_lambda 0.3 --use_logger --logger_filename casieAll-nnCoref-840d-MaxSen50-MaxSpan20-lambda0.3-K50 --save_checkpoint --checkpoint_name casieAll-0430-nnCoref-840d-MaxSen50-MaxSpan20-lambda0.3-K50-{epoch:02d}-{valid_avg_f1:.2f} --logs_path /home/featurize/Logs

+ For the bert-base Model:
    python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_bert_base.pkl --model bertCorefModel --bert_based --bert_name bert-base --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 20 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-bertbase-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan20-K50 --save_checkpoint --checkpoint_name casieAll-0430-bertModel-bertbase-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan20-K50-{epoch:02d}-{valid_avg_f1:.2f} --logs_path /home/featurize/Logs

+ For the spanBert Model:
    python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_spanbert_base.pkl --model bertCorefModel --bert_based --bert_name spanbert-base --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 20 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-spanbertbase-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan20-K50 --save_checkpoint --checkpoint_name casieAll-0430-bertModel-spanbertbase-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan20-K50-{epoch:02d}-{valid_avg_f1:.2f} --logs_path /home/featurize/Logs

    + Expriments for POS&Deprel
        python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_cyber.pkl --model cyberCorefModel --bert_based --bert_name spanbert-base --pd_solution sum --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 15 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-spanbertbase-pdsum-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50 --save_checkpoint --checkpoint_name casieAll-0430-bertModel-spanbertbase-pdsum-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50-{epoch:02d}-{valid_avg_f1:.2f} --logs_path /home/featurize/Logs

        python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_cyber.pkl --model cyberCorefModel --bert_based --bert_name spanbert-base --pd_solution mean --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 15 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-spanbertbase-pdmean-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50 --save_checkpoint --checkpoint_name casieAll-0430-bertModel-spanbertbase-pdmean-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50-{epoch:02d}-{valid_avg_f1:.2f} --logs_path /home/featurize/Logs

        python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_cyber.pkl --model cyberCorefModel --bert_based --bert_name spanbert-base --pd_solution lstm --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 15 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-spanbertbase-pdlstm-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50 --save_checkpoint --checkpoint_name casieAll-0430-bertModel-spanbertbase-pdlstm-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50-{epoch:02d}-{valid_avg_f1:.2f} --logs_path /home/featurize/Logs

        python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_cyber.pkl --model cyberCorefModel --bert_based --bert_name spanbert-base --pd_solution attn --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 15 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-spanbertbase-pdattn-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50 --save_checkpoint --checkpoint_name casieAll-0430-bertModel-spanbertbase-pdattn-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50-{epoch:02d}-{valid_avg_f1:.2f} --logs_path /home/featurize/Logs

    + Experiments for sents pair relation:
        python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_cyber.pkl --model cyberCorefModel --bert_based --bert_name spanbert-base --sent_corelation dattn --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 15 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-spanbertbase-srdattn-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50  --logs_path /home/featurize/Logs
    
    + Experiment for entity type:
        python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_cyber.pkl --model cyberCorefModel --bert_based --bert_name spanbert-base --insertTag --tp_solution tag --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 20 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-spanbertbase-tptag-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50  --logs_path /home/featurize/Logs

        without-tag-md:
        python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_cyber.pkl --model cyberCorefModel --bert_based --bert_name spanbert-base --tp_solution without-tag-md --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 20 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-spanbertbase-tp-nontag-md-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50  --logs_path /home/featurize/Logs --save_checkpoint --checkpoint_name casieAll-0430-bertModel-spanbertbase-tp-nontag-md-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50-{epoch:02d}-{valid_avg_f1:.2f}

        gold:
        python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_cyber.pkl --model cyberCorefModel --bert_based --bert_name spanbert-base --tp_solution gold --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 20 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-spanbertbase-tp-gold-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50  --logs_path ./Logs --save_checkpoint --checkpoint_name casieAll-0430-bertModel-spanbertbase-tp-gold-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50-{epoch:02d}-{valid_avg_f1:.2f}

+ For the corefBERT Model:
    python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_corefbert_base.pkl --model bertCorefModel --bert_based --bert_name corefbert-base --segment_max_num 1 --max_epochs 50 --scheduler CosineAnnealingLR --scheduler_T_max 15 --max_span_length 20 --prune_lambda 0.3 --use_logger --logger_filename casieAll-0430-bertModel-corefbertbase-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan20-K50 --save_checkpoint --checkpoint_name casieAll-0430-bertModel-corefbertbase-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan20-K50-{epoch:02d}-{valid_avg_f1:.2f} --logs_path /home/featurize/Logs

+ For the word-level Model:
    python train.py --corpus_subpath casieAll_0430 --corpus_filename _corpus_wordLevel.pkl --model wordLevelModel --bert_based --bert_name spanbert-base --segment_max_num 1 --max_epochs 50 --max_span_length 20 --prune_lambda 0.3 --scheduler None --use_logger --logger_filename casieAll-0430-wordLevel-spanbertbase-2lr-lambda0.3-MaxSeg1-Seglen384-MaxSpan20-K50 --save_checkpoint --checkpoint_name casieAll-0430-wordLevel-spanbertbase-2lr-lambda0.3-MaxSeg1-Seglen384-MaxSpan20-K50-{epoch:02d}-{valid_avg_f1:.2f} --logs_path /home/featurize/Logs

"""

args = arg_parse()

# Set the Dataset
train_corpus_path = args.dataset_path + args.corpus_subpath + '/train' + args.corpus_filename
val_corpus_path = args.dataset_path + args.corpus_subpath + '/val' + args.corpus_filename
train_corpus = load_corpus(train_corpus_path)
val_corpus = load_corpus(val_corpus_path)


# Set the DataLoader
n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_corpus, batch_size=None, batch_sampler=None, shuffle=False, num_workers=n_cpu)
valid_dataloader = DataLoader(val_corpus, batch_size=None, batch_sampler=None, shuffle=False, num_workers=n_cpu)

# Get Max Sentence Length
MaxSentLen = 0
for doc in train_corpus:
    cur_MaxSentLen = max([bdry[1]-bdry[0]+1 for bdry in doc.sent2subtok_bdry])
    if MaxSentLen < cur_MaxSentLen:
        MaxSentLen = cur_MaxSentLen
for doc in val_corpus:
    cur_MaxSentLen = max([bdry[1]-bdry[0]+1 for bdry in doc.sent2subtok_bdry])
    if MaxSentLen < cur_MaxSentLen:
        MaxSentLen = cur_MaxSentLen

# Select the Model
if args.model == 'nnCorefModel':
    model = nnCorefModel()
elif args.model == 'cyberCorefModel':
    model = cyberCorefModel(MaxSentLen)
elif args.model == 'bertCorefModel':
    model = bertCorefModel(distribute_model=args.distribute_model)
elif args.model == 'wordLevelModel':
    model = wordLevelModel(len(train_corpus))


# Set Checkpoint Callback
if args.save_checkpoint:
    checkpoint_callback = ModelCheckpoint(
        monitor=args.checkpoint_moniter_metirc,
        dirpath=args.checkpoint_path,
        filename=args.checkpoint_name,
        # save_top_k=3,
        mode="max",
    )
    callbacks = [checkpoint_callback]
else:
    callbacks = []

# Early Stop Callback
# early_stop_callback = EarlyStopping(monitor="train_avg_f1", min_delta=0.00001, patience=10, verbose=False, mode="max")
# callbacks.append(early_stop_callback)

# Set the Trainer
if args.use_logger:
    wandb_logger = WandbLogger(project="CyberCoref",
                            name = args.logger_filename,
                            save_dir = args.logs_path,
                            log_model="all")
                            
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=args.max_epochs,
        # strategy='ddp',
        logger = wandb_logger,
        callbacks=callbacks,
    )
else:
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=args.max_epochs,
        # strategy='ddp',
        callbacks=callbacks,
    )


# Start Training
trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader,
)