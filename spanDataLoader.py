import os
import json
from config import arg_parse
from dataLoader import load_corpus, Corpus, Document, BERTDocument
from Models import bertCorefModel

import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.utils.data import DataLoader, random_split

import pickle
from tqdm import tqdm

args = arg_parse()

def getSpansFromModel():
    """
    Retrive train samples from Model's Mention Detection Moudle
    Take Joshi et al. 2019b as baseline model 
    which achieves 0.885 and 0.270 of Recall and Precision on the MD task.
    """

    # Set the Dataset
    train_corpus_path = args.dataset_path + args.corpus_subpath + '/train' + args.corpus_filename
    val_corpus_path = args.dataset_path + args.corpus_subpath + '/val' + args.corpus_filename
    corpora = [load_corpus(train_corpus_path), load_corpus(val_corpus_path)]
    # train_corpus = load_corpus(train_corpus_path)
    # val_corpus = load_corpus(val_corpus_path)

    # Select Checkpoint Path
    checkpoint_path = args.checkpoint_path + '/' + args.load_checkpoint_name

    # Select Model
    model = bertCorefModel(distribute_model=args.distribute_model).load_from_checkpoint(checkpoint_path).to(args.device)
    
    span_samples={}
    doc_id = 0

    model.eval()
    for corpus in corpora:
        for doc in tqdm(corpus):
            # true mentions
            cur_ture_mentions = []
            cur_ture_mentions_for_check = []
            sent_offsets = [sum([len(j) for j in doc.sents[0:i]]) for i in range(len(doc.sents))]
            for entity in doc.entities:
                try:
                    word_from, word_to = entity[5][0], entity[5][-1]
                    # 以防span跨句的情况
                    if doc.word2sent[word_from] != doc.word2sent[word_to]:
                        continue
                    sent_id = doc.word2sent[word_from]
                    sent_offset = sent_offsets[sent_id]
                    content = doc.tokens[word_from:word_to+1]
                    type = entity[1]

                    span_sample = genNewSpanSample(sent_id, content, sent_offset, word_from, word_to, type)

                    cur_ture_mentions.append(span_sample)
                    cur_ture_mentions_for_check.append((sent_id, word_from, word_to))
                except:
                    continue

            # spans selected by model
            cur_span_samples = []
            spans, _ = model.forward(doc)
            for span in spans:
                try:
                    word_from, word_to = span.i1, span.i2
                    # 以防span跨句的情况
                    if doc.word2sent[word_from] != doc.word2sent[word_to]:
                        continue
                    sent_id = doc.word2sent[word_from]
                    sent_offset = sent_offsets[sent_id]

                    # 避免span重复
                    if (sent_id, word_from, word_to) in cur_ture_mentions_for_check:
                        continue

                    content = span.content
                    type = 'None'

                    span_sample = genNewSpanSample(sent_id, content, sent_offset, word_from, word_to, type)
                    cur_span_samples.append(span_sample)
                except:
                    continue
            
            span_samples[doc_id] = {
                'sents': dict(zip(range(len(doc.sents)),doc.sents)),
                'cur_ture_mentions': cur_ture_mentions,
                'cur_span_samples': cur_span_samples
            }
            doc_id +=1


    with open('./Dataset/span_samples.json', 'w') as fs:
        span_samples = json.dumps(span_samples,indent=4)
        fs.write(span_samples)


def genNewSpanSample(sent_id, content, sent_offset, word_from, word_to, type):
    return {
        'sent_id':sent_id,
        'content': content,
        'word_from':word_from-sent_offset,
        'word_to':word_to-sent_offset,
        'type':type
    }

if args.bert_based:
    global bert_tokenizer
    # bert_tokenizer = BertTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    bert_tokenizer = BertTokenizer.from_pretrained("./BERTs/spanbert-base-cased")
    if args.insertTag:
        bert_tokenizer.add_tokens("<SST>", special_tokens=True)
        bert_tokenizer.add_tokens("<SND>", special_tokens=True)

class candidateMentions:
    def __init__(self, span_samples):
        self.spans = span_samples,
        self.spans = self.spans[0]
        if args.insertTag:
            self._insertTag()
        self._bert_tokenize()
        self._type2label()

    def __getitem__(self, idx):
        return self.spans[idx]

    def __len__(self):
        return len(self.spans)
    
    def _insertTag(self):
        start_tag = ["<SST>"]
        end_tag = ["<SND>"]

        for i, span in enumerate(self.spans):
            self.spans[i]['tagged_sent'] = span['sent'][:span['word_from']]+start_tag+span['sent'][span['word_from']:span['word_to']+1]+end_tag+span['sent'][span['word_to']+1:]

            self.spans[i]['word_from'] = span['word_from'] + 1
            self.spans[i]['word_to'] = span['word_to'] + 1
    
    def _bert_tokenize(self):
        for i, span in tqdm(enumerate(self.spans)):
            word2token = {}
            bert_tokens = [101]
            
            if args.insertTag:
                for j, word in enumerate(span['tagged_sent']):
                    tokens = bert_tokenizer(word)['input_ids'][1:-1]

                    word2token[j]= list(range(len(bert_tokens), len(bert_tokens)+len(tokens)))
                    bert_tokens.extend(tokens)
            else:
                for j, word in enumerate(span['sent']):
                    tokens = bert_tokenizer(word)['input_ids'][1:-1]

                    word2token[j]= list(range(len(bert_tokens), len(bert_tokens)+len(tokens)))
                    bert_tokens.extend(tokens)
            bert_tokens = torch.LongTensor(bert_tokens+[102])

            self.spans[i]['bert_tokens'] = bert_tokens
            self.spans[i]['word2token'] = word2token
            self.spans[i]['token_from'] = word2token[span['word_from']][0]
            self.spans[i]['token_to'] = word2token[span['word_to']][-1]
    
    def _type2label(self):
        with open('./Dataset/others/type2label.json', 'r') as fs:
            type2label = json.load(fs)

        for i, span in enumerate(self.spans):
            self.spans[i]['label'] = type2label[span['type']]


def collate_fn(data):
    labels = [span['label'] for span in data]
    spans = [(span['token_from'], span['token_to']) for span in data]
    sents = [span['bert_tokens'] for span in data]

    batched_sents = pad_sequence(sents, batch_first=True)

    return batched_sents, spans, labels



if __name__ == "__main__":
    # getSpansFromModel()

    with open('./Dataset/candidateMentions/span_samples.json', 'r') as fs:
        span_dataset = json.load(fs)

    span_samples = []
    for doc in span_dataset.values():
        cur_samples = doc['cur_ture_mentions'] + doc['cur_span_samples']
        for i, span in enumerate(cur_samples):
            cur_samples[i]['sent'] = doc['sents'][str(span['sent_id'])]
        
        span_samples += cur_samples
    
    dataset = candidateMentions(span_samples)

    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.2)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    pickle.dump(train_dataset, open('./Dataset/candidateMentions/train_candidate_mentions_nontag.pkl', 'wb'))
    pickle.dump(val_dataset, open('./Dataset/candidateMentions/val_candidate_mentions_nontag.pkl', 'wb'))






