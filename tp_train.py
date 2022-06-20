from spanDataLoader import candidateMentions, collate_fn
from dataLoader import load_corpus, Corpus
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import pickle
from config import arg_parse
from Models import typePredModel
args = arg_parse()

"""
    python tp_train.py --max_epochs 20 --tp_solution tag --use_logger --logger_filename tp-spanbert-tagged-tag-new --save_checkpoint --checkpoint_moniter_metirc valid_weighted_f1 --checkpoint_name  tp-spanbert-tagged-tag-new-{epoch:02d}-{valid_weighted_f1:.2f}
    
    python tp_train.py --max_epochs 20 --tp_solution tagged-mean --use_logger --logger_filename tp-spanbert-tagged-mean --save_checkpoint --checkpoint_moniter_metirc valid_weighted_f1 --checkpoint_name  tp-spanbert-tagged-mean-{epoch:02d}-{valid_weighted_f1:.2f}
    
    python tp_train.py --max_epochs 15 --tp_solution tag --use_logger --logger_filename tp-spanbert-tagged-tag-32 --save_checkpoint --checkpoint_moniter_metirc valid_weighted_f1 --checkpoint_name  tp-spanbert-tagged-tag-32-{epoch:02d}-{valid_weighted_f1:.2f}
    
    python tp_train.py --max_epochs 20 --tp_solution without-tag-md --use_logger --logger_filename tp-spanbert-without-tag-md --model bertCorefModel --save_checkpoint --checkpoint_moniter_metirc valid_weighted_f1 --checkpoint_name  tp-spanbert-without-tag-md-{epoch:02d}-{valid_weighted_f1:.2f}
    
    python tp_train.py --max_epochs 20 --tp_solution without-tag-mean --use_logger --logger_filename tp-spanbert-without-tag-mean --model bertCorefModel --save_checkpoint --checkpoint_moniter_metirc valid_weighted_f1 --checkpoint_name  tp-spanbert-without-tag-mean-{epoch:02d}-{valid_weighted_f1:.2f}
    
    python tp_train.py --max_epochs 20 --tp_solution tagged-token-md --use_logger --logger_filename tp-spanbert-tagged-md-new --save_checkpoint --checkpoint_moniter_metirc valid_weighted_f1 --checkpoint_name  tp-spanbert-tagged-md-new-{epoch:02d}-{valid_weighted_f1:.2f} --device cuda:0 --gpus 0
    
    python tp_train.py --max_epochs 20 --insertTag --tp_solution tag --use_logger --logger_filename tp-spanbert-tagged-tag-new2 --save_checkpoint --checkpoint_moniter_metirc valid_weighted_f1 --checkpoint_name  tp-spanbert-tagged-tag-new2-{epoch:02d}-{valid_weighted_f1:.2f} --device cuda:0 --gpus 0

    python tp_train.py --max_epochs 20 --insertTag --tp_solution tagged-token-md --use_logger --logger_filename tp-spanbert-tagged-md-new2 --save_checkpoint --checkpoint_moniter_metirc valid_weighted_f1 --checkpoint_name  tp-spanbert-tagged-md-new2-{epoch:02d}-{valid_weighted_f1:.2f} --device cuda:0 --gpus 0
"""
train_dataset = pickle.load(open("./Dataset/candidateMentions/train_candidate_mentions.pkl", 'rb'))
val_dataset = pickle.load(open("./Dataset/candidateMentions/val_candidate_mentions.pkl", 'rb'))
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=16, shuffle=True)

model = typePredModel()

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

# Set the Trainer
if args.use_logger:
    wandb_logger = WandbLogger(project="TypePredict",
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
        callbacks=callbacks,
    )


# Start Training
trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader,
)