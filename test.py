import os
from config import arg_parse
from dataLoader import load_corpus, Corpus, Document, BERTDocument, corefQADocument, wordLevelDocument
from Models import bertCorefModel, nnCorefModel, wordLevelModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import extract_gold_coref_cluster, extract_pred_coref_cluster
from torch.utils.data import DataLoader

"""
To run the test.py, you should set the following arguments:
1/corpus_filename
2/model
3/load_checkpoint_name
"""

args = arg_parse()

# Set the Dataset

val_corpus_path = args.dataset_path + args.corpus_subpath + '/val' + args.corpus_filename
val_corpus = load_corpus(val_corpus_path)
test_corpus = val_corpus


# Set the DataLoader
# n_cpu = os.cpu_count()
# test_dataloader = DataLoader(val_corpus, batch_size=None, batch_sampler=None, shuffle=False, num_workers=n_cpu)


# Set Checkpoint Path
checkpoint_path = args.checkpoint_path + '/' + args.load_checkpoint_name

# Select the Model
if args.model == 'nnCorefModel':
    model = nnCorefModel().load_from_checkpoint(checkpoint_path)
elif args.model == 'bertCorefModel':
    model = bertCorefModel(distribute_model=args.distribute_model).load_from_checkpoint(checkpoint_path).to(args.device)
elif args.model == 'wordLevelModel':
    model = wordLevelModel(len(test_corpus)).load_from_checkpoint(checkpoint_path).to(args.device)

model.eval()
doc = test_corpus[args.probe_doc_idx]

if args.model == 'wordLevelModel':  
    result = model(doc)

    ### Check mentions head word Selection
    print("-"*20 + "Check Mentions'Head Word Selection" + "-"*20)
    for idx, item in enumerate(doc.head2span):
        print("idx: %d, Mention: %s, Headword: %s" % (idx, doc.cased_words[item[1]:item[2]], doc.cased_words[item[0]]))


    ### Check Word-Level Corefence Resolution
    print("-"*20 + "Check Word-Level Corefence Resolution" + "-"*20)

    for idx, cluster in enumerate(result.word_clusters):
        print("-"*10 + "\033[34mThe %dth Predict Cluster:\033[0m" % idx +"-"*10)
        for word_id in cluster:
            print("%d: %s\t" % (word_id, doc.cased_words[word_id]))

    for idx, cluster in enumerate(doc.word_clusters):
        print("-"*10 + "\033[33mThe %dth Gold Cluster:\033[0m" % idx +"-"*10)
        for word_id in cluster:
            print("%d: %s\t" % (word_id, doc.cased_words[word_id]))


    ### Check the Word2Span Prediction
    print("-"*20 + "Check Word2Span Corefence Resolution" + "-"*20)
    print("Correct predicted head word and its predicted span:")
    gold_words = [word for cluster in doc.word_clusters for word in cluster]

    for word_cluster, span_cluster in zip(result.word_clusters, result.span_clusters):
        for word, span in zip(word_cluster, span_cluster):
            if word in gold_words:
                gold_span = [doc.cased_words[head[1]:head[2]] for head in doc.head2span if head[0] == word][0]
                print("%s: %s: %s" % (word, doc.cased_words[span[0]:span[1]], gold_span))


if args.model == 'bertCorefModel':
    
    spans, scores = model.forward(doc)
    
    gold_coref_cluster = extract_gold_coref_cluster(doc)
    pred_coref_cluster = extract_pred_coref_cluster(spans, scores)

    ### Check Mention Selection
    print("-"*20 + "Check Mentions Selection" + "-"*20)

    gold_mentions = [coref['span'] for coref in doc.corefs]
    pred_mentions = [(span.i1, span.i2) for span in spans]
    for idx, span in enumerate(spans):
        if (span.i1, span.i2) in gold_mentions:
            print("\033[32m%d: (%s, %s)\t%s\033[0m" % (idx, span.i1, span.i2, doc.tokens[span.i1:span.i2+1]))
        else:
            print("\033[31m%d: (%s, %s)\t%s\033[0m" % (idx, span.i1, span.i2, doc.tokens[span.i1:span.i2+1]))
    
    print('-'*20+"Unpredicted Spans:"+'-'*20)
    for span in gold_mentions:
        if span not in pred_mentions:
            print("\033[34m%d: (%s, %s)\t%s\033[0m" % (idx, span[0], span[1], doc.tokens[span[0]:span[1]+1]))
    

    ### Check Span Coreference Resolution
    print("-"*20 + "Check Span Coreference Resolution" + "-"*20)

    for idx, cluster in enumerate(pred_coref_cluster):
        print("-"*10 + "\033[34mThe %dth Predict Cluster:\033[0m" % idx +"-"*10)
        for span in cluster:
            print("(%d, %d)\t%s" % (span[0], span[1], doc.tokens[span[0]:span[1]+1]))

    for idx, cluster in enumerate(gold_coref_cluster):
        print("-"*10 + "\033[33mThe %dth Gold Cluster:\033[0m" % idx +"-"*10)
        for span in cluster:
            print("(%d, %d)\t%s" % (span[0], span[1], doc.tokens[span[0]:span[1]+1]))