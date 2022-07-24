import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    """
    Be careful with the following configurations: 
    --model
    --bert_based
    --bert_name
    --train_corpus_path
    --val_corpus_path
    """

    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--distribute_model', action='store_true', default=False, 
                        help='Whether or not to spread the model across 3 GPUs')
    parser.add_argument('--train', action='store_true', default=True, help='Train model')
    parser.add_argument('--test', action='store_true', default=False, help='Test model')
    parser.add_argument('--probe_doc_idx', default=1, type=int, help='Test model')
    parser.add_argument('--dict_path', default='./Dataset/others/')

    # Dataset Path
    parser.add_argument('--dataset_path', default='./Dataset/')
    parser.add_argument('--corpus_subpath', default='casieAll_0430')
    parser.add_argument('--corpus_filename', default='_corpus_cyber.pkl')
    parser.add_argument('--corpus_len', default=100)
    parser.add_argument('--max_sent_len', default=179)
    # train_corpus_path = ./Dataset/casie0417/train_corpus_bert_base.pkl
    # val_corpus_path = ./Dataset/casie0417/val_corpus_bert_base.pkl
    
    # Embedding Weights
    parser.add_argument('--embedding_weights_path', default='./Weights/embeddingWeights/.vector_cache/')

    # Model
    parser.add_argument('--model', default='cyberCorefModel', choices = ['bertCorefModel', 'nnCorefModel', 'corefQAModel', 'wordLevelModel', 'cyberCorefModel'], help='Select a Model first.')
    parser.add_argument('--bert_based', action='store_true', default=True, help='')
    parser.add_argument('--bert_name', default='spanbert-base', choices = ['bert-base', 'bert-large', 'spanbert-base', 'spanbert-large', 'corefbert-base', 'corefbert-large', 'corefroberta-base', 'corefroberta-large'], help='Select a bert version.')

    parser.add_argument('--freeze_bert', action='store_true', default=False, help='')
    parser.add_argument('--freeze_embeds', action='store_true', default=False)
    parser.add_argument('--hidden_dim', default=512, type=int, help='hidden size')
    parser.add_argument('--atten_dim', default=768, type=int, help='attention weight size')
    parser.add_argument('--embeds_dim', default=768, type=int, help='')
    parser.add_argument('--distance_dim', default=64, type=int, help='')
    parser.add_argument('--pos_dim', default=64, type=int, help='')
    parser.add_argument('--deprel_dim', default=64, type=int, help='')
    parser.add_argument('--width_dim', default=32, type=int, help='')
    parser.add_argument('--genre_dim', default=20, type=int, help='')
    parser.add_argument('--speaker_dim', default=20, type=int, help='')
    parser.add_argument('--cnn_char_filters', default=50, type=int, help='')
    parser.add_argument('--dropout_rate', default=0.3, help='')
    
    # wordLevelModel Specific
    parser.add_argument('--wl_ascoring_batch_size', default=512, help='')
    parser.add_argument('--wl_anaphoricity_hidden_size', default=1024)
    parser.add_argument('--wl_n_hidden_layer', default=1)
    parser.add_argument('--wl_feature_embeds_dim', default=20)
    parser.add_argument('--wl_bce_loss_weight', default=0.5)
    parser.add_argument('--wl_bert_window_size', default=512)
    parser.add_argument('--wl_bert_learning_rate', default=1e-5)

    # Expriments for adding POS and Deprel
    parser.add_argument('--pd_solution', default='none', choices = ['sum', 'mean', 'lstm', 'attn', 'none'])
    parser.add_argument('--mention_coref_gi_split', default=False, action='store_true')

    # Expriments for sents corelation
    parser.add_argument('--sent_corelation', default='none', choices = ['lstm','dattn','dot-matchPyramid', 'cos-matchPyramid', 'cos-dot-matchPyramid', 'none'])
    parser.add_argument('--sent_corel_dim', default=32)

    # Experiments for type prediction
    parser.add_argument('--tp_all_in_one', action='store_true', default=True)
    parser.add_argument('--insertTag', action='store_true', default=False)
    parser.add_argument('--type_dim', default=64, type=int, help='')
    parser.add_argument('--tp_solution', default='without-tag-md', choices = ['without-tag-mean', 'without-tag-md', 'tagged-mean', 'tagged-token-md', 'tag', 'gold', 'None'])

    # Train
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--train_val_ratio', default=0.2, type=float)
    parser.add_argument('--max_segment_len', default=384, type=int)
    parser.add_argument('--segment_max_num', default=1, type=int, help='')
    parser.add_argument('--sentense_max_num', default=60, type=int, help='')
    parser.add_argument('--prune_lambda', default=0.3, type=float)
    parser.add_argument('--max_span_length', default=20, type=int, help='')
    parser.add_argument('--top_K', default=50, type=int, help='')
    parser.add_argument('--higer_order_N', default=1, type=int, help='')
    parser.add_argument('--lr', default=3e-4, help='')
    parser.add_argument('--bert_lr', default=1e-5, help='')
    parser.add_argument('--scheduler', default='ExponentialLR', choices=['ExponentialLR', 'CosineAnnealingLR', 'None'])
    parser.add_argument('--scheduler_gamma', default=0.9, help='')
    parser.add_argument('--scheduler_T_max', default=15, help='', type=int)
    
    
    # Checkpoint
    parser.add_argument('--save_checkpoint', action='store_true', default=False)
    parser.add_argument('--checkpoint_path', default='./Weights')
    parser.add_argument('--checkpoint_moniter_metirc', default='valid_avg_f1')
    parser.add_argument('--checkpoint_name', default='casie100-bertbasecoref-{epoch:02d}-{valid_avg_f1:.2f}')
    parser.add_argument('--load_checkpoint_name', default='casieAll-0430-bertModel-spanbertbase-tp-allinone-pretrained-nontag-md-2lr-lambda0.3-ca15-MaxSeg1-Seglen384-MaxSpan15-K50-epoch=33-valid_avg_f1=0.61.ckpt')

    # Logger
    parser.add_argument('--use_logger', action='store_true', default=False)
    parser.add_argument('--logs_path', default='./Logs')
    parser.add_argument('--logger_filename', default='bertbase-casie100-MaxSeg1-MaxSpan30-K50')


    # Trained Model 
    parser.add_argument('--pretrained_coref_path', default=None, help='Path to pretrained model')


    args = parser.parse_args()

    return args