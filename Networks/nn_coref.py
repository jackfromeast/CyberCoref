"""
End-to-end Neural Coreference Resolution 
"""
import sys 
sys.path.append("..") 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.vocab import Vectors
import attr
from utils import *
from dataLoader import Span
from .embedding import LazyVectors
from dataLoader import load_corpus

from config import arg_parse
args = arg_parse()

global train_corpus, val_corpus, DEVICE
train_corpus_path = args.dataset_path + args.corpus_subpath + '/train' + args.corpus_filename
val_corpus_path = args.dataset_path + args.corpus_subpath + '/val' + args.corpus_filename
train_corpus = load_corpus(train_corpus_path)
val_corpus = load_corpus(val_corpus_path)

DEVICE = 'cuda:0' if args.gpus == 1 else 'cpu'

def setupEmbeddings():
    GLOVE = LazyVectors.from_corpus(train_corpus.vocab,
                                    name='glove.6B.300d.txt',
                                    cache=args.embedding_weights_path)

    TURIAN = LazyVectors.from_corpus(train_corpus.vocab,
                                    name='hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt',
                                    cache=args.embedding_weights_path)
    
    return GLOVE, TURIAN

def lookup_tensor(tokens, vectorizer):
    """ Convert a sentence to an embedding lookup tensor """
    return torch.tensor([vectorizer.stoi(t) for t in tokens]).to(DEVICE)

class Score(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim, hidden_dim=args.hidden_dim):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)


class Distance(nn.Module):
    """ Learned, continuous representations for: distance
    between spans
    """

    bins = torch.cuda.LongTensor([1,2,3,4,8,16,32,64])

    def __init__(self, distance_dim=20):
        super().__init__()

        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, distance_dim),
            nn.Dropout(0.3)
        )

    def forward(self, *args):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def stoi(self, lengths):
        """ Find which bin a number falls into """
        return torch.tensor([
            sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False
        ).to(DEVICE)


class Width(nn.Module):
    """ Learned, continuous representations for: span width
    """
    
    def __init__(self, L, width_dim=args.width_dim):
        super().__init__()

        self.dim = width_dim
        self.embeds = nn.Sequential(
            nn.Embedding(L, width_dim),
        )

    def forward(self, widths):
        """ Embedding table lookup """
        return self.embeds(widths)


class Genre(nn.Module):
    """ Learned continuous representations for genre. Zeros if genre unknown.
    """

    genres = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
    _stoi = {genre: idx+1 for idx, genre in enumerate(genres)}

    def __init__(self, genre_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(len(self.genres)+1, genre_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, labels):
        """ Embedding table lookup """
        return self.embeds(self.stoi(labels))

    def stoi(self, labels):
        """ Locate embedding id for genre """
        indexes = [self._stoi.get(gen) for gen in labels]
        return torch.tensor([i if i is not None else 0 for i in indexes]).to(DEVICE)


class Speaker(nn.Module):
    """ Learned continuous representations for binary speaker. Zeros if speaker unknown.
    """

    def __init__(self, speaker_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(3, speaker_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, speaker_labels):
        """ Embedding table lookup (see src.utils.speaker_label fnc) """
        return self.embeds(torch.tensor(speaker_labels).to(DEVICE))


class CharCNN(nn.Module):
    """ Character-level CNN. Contains character embeddings.
    """

    unk_idx = 1
    vocab = train_corpus.char_vocab
    _stoi = {char: idx+2 for idx, char in enumerate(vocab)}
    pad_size = 15

    def __init__(self, filters, char_dim=8):
        super().__init__()

        self.embeddings = nn.Embedding(len(self.vocab)+2, char_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.pad_size,
                                              out_channels=filters,
                                              kernel_size=n) for n in (3,4,5)])

    def forward(self, sent):
        """ Compute filter-dimensional character-level features for each doc token """
        embedded = self.embeddings(self.sent_to_tensor(sent))
        convolved = torch.cat([F.relu(conv(embedded)) for conv in self.convs], dim=2)
        pooled = F.max_pool1d(convolved, convolved.shape[2]).squeeze(2)
        return pooled

    def sent_to_tensor(self, sent):
        """ Batch-ify a document class instance for CharCNN embeddings """
        tokens = [self.token_to_idx(t) for t in sent]
        batch = self.char_pad_and_stack(tokens)
        return batch

    def token_to_idx(self, token):
        """ Convert a token to its character lookup ids """
        return torch.tensor([self.stoi(c) for c in token]).to(DEVICE)

    def char_pad_and_stack(self, tokens):
        """ Pad and stack an uneven tensor of token lookup ids """
        skimmed = [t[:self.pad_size] for t in tokens]

        lens = [len(t) for t in skimmed]

        padded = [F.pad(t, (0, self.pad_size-length))
                  for t, length in zip(skimmed, lens)]

        return torch.stack(padded)

    def stoi(self, char):
        """ Lookup char id. <PAD> is 0, <UNK> is 1. """
        idx = self._stoi.get(char)
        return idx if idx else self.unk_idx


class DocumentEncoder(nn.Module):
    """ Document encoder for tokens
    """
    def __init__(self, hidden_dim, char_filters, n_layers=2):
        super().__init__()

        self.GLOVE, self.TURIAN = setupEmbeddings()

        # Unit vector embeddings as per Section 7.1 of paper
        glove_weights = F.normalize(self.GLOVE.weights())
        turian_weights = F.normalize(self.TURIAN.weights())

        # GLoVE
        self.glove = nn.Embedding(glove_weights.shape[0], glove_weights.shape[1])
        self.glove.weight.data.copy_(glove_weights)
        if args.freeze_embeds:
            self.glove.weight.requires_grad = False

        # Turian
        self.turian = nn.Embedding(turian_weights.shape[0], turian_weights.shape[1])
        self.turian.weight.data.copy_(turian_weights)
        if args.freeze_embeds:
            self.turian.weight.requires_grad = False


        # Character
        self.char_embeddings = CharCNN(char_filters)

        # Sentence-LSTM
        self.lstm = nn.LSTM(glove_weights.shape[1]+turian_weights.shape[1]+char_filters,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout
        self.emb_dropout = nn.Dropout(0.40)
        self.lstm_dropout = nn.Dropout(0.20)

    def forward(self, doc):
        """ Convert document words to ids, embed them, pass through LSTM. """

        # Embed document
        embeds = [self.embed(s) for s in doc.sents]

        # Batch for LSTM
        packed, reorder = pack(embeds)

        # Apply embedding dropout
        self.emb_dropout(packed[0])

        # Pass an LSTM over the embeds
        output, _ = self.lstm(packed)

        # Apply dropout
        self.lstm_dropout(output[0])

        # Undo the packing/padding required for batching
        states = unpack_and_unpad(output, reorder)

        return torch.cat(states, dim=0), torch.cat(embeds, dim=0)

    def embed(self, sent):
        """ Embed a sentence using GLoVE, Turian, and character embeddings """

        # Embed the tokens with Glove
        glove_embeds = self.glove(lookup_tensor(sent, self.GLOVE))

        # Embed again using Turian this time
        tur_embeds = self.turian(lookup_tensor(sent, self.TURIAN))

        # Character embeddings
        char_embeds = self.char_embeddings(sent)

        # Concatenate them all together
        embeds = torch.cat((glove_embeds, tur_embeds, char_embeds), dim=1)

        return embeds


class MentionScore(nn.Module):
    """ Mention scoring module
    """
    def __init__(self, gi_dim, attn_dim, distance_dim):
        super().__init__()

        self.attention = Score(attn_dim)
        self.width = Width(args.max_span_length, args.width_dim)
        self.score = Score(gi_dim)

    def forward(self, states, embeds, doc, K=args.top_K):
        """ Compute unary mention score for each span
        """

        # Initialize Span objects containing start index, end index, genre, speaker
        spans = [Span(i1=i[0], i2=i[-1], id=idx,
                      speaker=doc.speaker(i), genre=doc.genre)
                 for idx, i in enumerate(compute_idx_spans(doc.sents))]

        # Compute first part of attention over span states (alpha_t)
        attns = self.attention(states)

        # Regroup attn values, embeds into span representations
        # TODO: figure out a way to batch
        span_attns, span_embeds = zip(*[(attns[s.i1:s.i2+1], embeds[s.i1:s.i2+1])
                                        for s in spans])

        # Pad and stack span attention values, span embeddings for batching
        padded_attns = pad_and_stack(span_attns, value=-1e10)
        padded_embeds = pad_and_stack(span_embeds)

        # Weight attention values using softmax
        attn_weights = F.softmax(padded_attns, dim=1)

        # Compute self-attention over embeddings (x_hat)
        attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)

        # Compute span widths (i.e. lengths), embed them
        widths = self.width(torch.tensor([len(s) for s in spans]).to(DEVICE))

        # Get LSTM state for start, end indexes
        # TODO: figure out a way to batch
        start_end = torch.stack([torch.cat((states[s.i1], states[s.i2]))
                                 for s in spans])

        # Cat it all together to get g_i, our span representation
        g_i = torch.cat((start_end, attn_embeds, widths), dim=1)

        # Compute each span's unary mention score
        mention_scores = self.score(g_i)

        # Update span object attributes
        # (use detach so we don't get crazy gradients by splitting the tensors)
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores.detach())
        ]

        # Prune down to LAMBDA*len(doc) spans
        spans = prune(spans, len(doc))

        # Update antencedent set (yi) for each mention up to K previous antecedents
        spans = [
            attr.evolve(span, yi=spans[max(0, idx-K):idx])
            for idx, span in enumerate(spans)
        ]

        return spans, g_i, mention_scores


class PairwiseScore(nn.Module):
    """ Coreference pair scoring module
    """
    def __init__(self, gij_dim, distance_dim, genre_dim, speaker_dim):
        super().__init__()

        self.distance = Distance(distance_dim)
        self.genre = Genre(genre_dim)
        self.speaker = Speaker(speaker_dim)

        self.score = Score(gij_dim)

    def forward(self, spans, g_i, mention_scores):
        """ Compute pairwise score for spans and their up to K antecedents
        """

        # Extract raw features
        mention_ids, antecedent_ids, \
            distances, genres, speakers = zip(*[(i.id, j.id,
                                                i.i2-j.i1, i.genre,
                                                speaker_label(i, j))
                                             for i in spans
                                             for j in i.yi])

        # For indexing a tensor efficiently
        mention_ids = torch.tensor(mention_ids).to(DEVICE)
        antecedent_ids = torch.tensor(antecedent_ids).to(DEVICE)

        # Embed them
        phi = torch.cat((self.distance(distances),
                         self.genre(genres),
                         self.speaker(speakers)), dim=1)

        # Extract their span representations from the g_i matrix
        i_g = torch.index_select(g_i, 0, mention_ids)
        j_g = torch.index_select(g_i, 0, antecedent_ids)

        # Create s_ij representations
        pairs = torch.cat((i_g, j_g, i_g*j_g, phi), dim=1)

        # Extract mention score for each mention and its antecedents
        s_i = torch.index_select(mention_scores, 0, mention_ids)
        s_j = torch.index_select(mention_scores, 0, antecedent_ids)

        # Score pairs of spans for coreference link
        s_ij = self.score(pairs)

        # Compute pairwise scores for coreference links between each mention and
        # its antecedents
        coref_scores = torch.sum(torch.cat((s_i, s_j, s_ij), dim=1), dim=1, keepdim=True)

        # Update spans with set of possible antecedents' indices, scores
        spans = [
            attr.evolve(span,
                        yi_idx=[((y.i1, y.i2), (span.i1, span.i2)) for y in span.yi]
                        )
            for span, score, (i1, i2) in zip(spans, coref_scores, pairwise_indexes(spans))
        ]

        # Get antecedent indexes for each span
        antecedent_idx = [len(s.yi) for s in spans if len(s.yi)]

        # Split coref scores so each list entry are scores for its antecedents, only.
        # (NOTE that first index is a special case for torch.split, so we handle it here)
        split_scores = [torch.tensor([]).to(DEVICE)] \
                         + list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = to_var(torch.tensor([[0.]]))
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        # Batch and softmax
        # get the softmax of the scores for each span in the document given
        probs = [F.softmax(tensr, dim=0) for tensr in with_epsilon]
        
        # pad the scores for each one with a dummy value, 1000 so that the tensors can 
        # be of the same dimension for calculation loss and what not. 
        probs = pad_and_stack(probs, value=-1e10)
        probs = probs.squeeze()
       
        return spans, probs


class NeuralCorefScorer(nn.Module):
    """ Super class to compute coreference links between spans
    """
    def __init__(self, embeds_dim=args.embeds_dim,
                       hidden_dim=args.hidden_dim,
                       char_filters=args.cnn_char_filters,
                       distance_dim=args.distance_dim,
                       genre_dim=args.genre_dim,
                       speaker_dim=args.speaker_dim):

        super().__init__()

        assert embeds_dim == 2*hidden_dim, "For End-to-End Model, the embeds_dim should match with 2*hidden_dim."

        # Forward and backward pass over the document
        attn_dim = hidden_dim*2

        # Forward and backward passes, avg'd attn over embeddings, span width
        gi_dim = attn_dim*2 + embeds_dim + distance_dim

        # gi, gj, gi*gj, distance between gi and gj
        gij_dim = gi_dim*3 + distance_dim + genre_dim + speaker_dim

        # Initialize modules
        self.encoder = DocumentEncoder(hidden_dim, char_filters)
        self.score_spans = MentionScore(gi_dim, attn_dim, distance_dim)
        self.score_pairs = PairwiseScore(gij_dim, distance_dim, genre_dim, speaker_dim)

    def forward(self, doc):
        """ Enocde document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.encoder(doc)

        # Get mention scores for each span, prune
        spans, g_i, mention_scores = self.score_spans(states, embeds, doc)

        # Get pairwise scores for each span combo
        spans, coref_scores = self.score_pairs(spans, g_i, mention_scores)

        return spans, coref_scores
