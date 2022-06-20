from dataLoader import load_corpus, Document, Corpus, wordLevelDocument
import jsonlines
from config import arg_parse

args = arg_parse()

def doc2jsonlines():
    train_corpus_path = args.dataset_path + args.corpus_subpath + '/train' + args.corpus_filename
    val_corpus_path = args.dataset_path + args.corpus_subpath + '/val' + args.corpus_filename
    train_corpus = load_corpus(train_corpus_path)
    val_corpus = load_corpus(val_corpus_path)
    
    train_corpus_jl = jsonlines.open('./train_corpus_jl.jsonlines', mode='a')
    for i in range(0, len(train_corpus)):
        cur_doc = train_corpus[i]
        
        cur_data = {
            'document_id': cur_doc.filename,
            'cased_words': cur_doc.cased_words,
            'sent_id': cur_doc.sent_id,
            'part_id': cur_doc.part_id,
            'speaker': cur_doc.speakers,
            'pos': cur_doc.pos,
            'deprel': cur_doc.deprel,
            'head': cur_doc.head,
            'head2span': cur_doc.head2span,
            'word_clusters': cur_doc.word_clusters,
            'span_clusters': cur_doc.span_clusters
        }

        train_corpus_jl.write(cur_data)
    train_corpus_jl.close()

    val_corpus_jl = jsonlines.open('./val_corpus_jl.jsonlines', mode='a')
    for i in range(0, len(val_corpus)):
        cur_doc = val_corpus[i]
        
        cur_data = {
            'document_id': cur_doc.filename,
            'cased_words': cur_doc.cased_words,
            'sent_id': cur_doc.sent_id,
            'part_id': cur_doc.part_id,
            'speaker': cur_doc.speakers,
            'pos': cur_doc.pos,
            'deprel': cur_doc.deprel,
            'head': cur_doc.head,
            'head2span': cur_doc.head2span,
            'word_clusters': cur_doc.word_clusters,
            'span_clusters': cur_doc.span_clusters
        }

        val_corpus_jl.write(cur_data)
    val_corpus_jl.close()


if __name__ == '__main__':
    doc2jsonlines()