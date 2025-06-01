import torch
from model import EncoderDecoder
import numpy as np

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
import yaml

with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
    DEVICE = torch.device('cuda')
else:
    from torch import FloatTensor, LongTensor
    DEVICE = torch.device('cpu')

def tokens_to_words(word_filed, tokens):
    return [word_filed.vocab.itos[token] for token in tokens]

def words_to_tokens(word_field, word):
    tokens = [BOS_TOKEN] + word_field.preprocess(word) + [EOS_TOKEN]
    return torch.tensor([word_field.vocab.stoi[token] for token in tokens])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask).to(DEVICE) == 0

def make_mask(source_inputs, target_inputs, pad_idx):
    source_mask = (source_inputs != pad_idx).unsqueeze(-2)
    target_mask = (target_inputs != pad_idx).unsqueeze(-2)
    target_mask = target_mask & subsequent_mask(target_inputs.size(-1)).type_as(target_mask)
    return source_mask, target_mask


def convert_batch(batch, pad_idx=1):
    source_inputs, target_inputs = batch.source.transpose(0, 1), batch.target.transpose(0, 1)
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx)

    return source_inputs, target_inputs, source_mask, target_mask

def decoding_tokens(tensor, vocab, skip_tokens=('<pad>', '<sos>', '<eos>')):
    sentences = []
    skip_ids = [vocab.stoi[token] for token in skip_tokens if token in vocab.stoi]

    for seq in tensor:
        tokens = [
            vocab.itos[token.item()]
            for token in seq
            if token.item() not in skip_ids
        ]
        sentences.append(" ".join(tokens))

    return sentences

def load_model(word_field, path, d_model=200,pretrained_embeddings=False):
    if d_model is None:
        d_model = 200
    filename = path.split('/')[-1].lower()
    if pretrained_embeddings is None:
        pretrained_embeddings = 'emb' in filename
    model = EncoderDecoder(
        source_vocab_size=len(word_field.vocab),
        target_vocab_size=len(word_field.vocab),
        d_model=d_model,
        pretrained_embeddings=pretrained_embeddings,
        word_field=word_field).to(DEVICE)
    model.eval()

    return model


def load_model_vis(word_field, path,d_model=200,pretrained_embeddingsА=False):
    if d_model is None:
        d_model = 200
    filename = path.split('/')[-1].lower()
    if pretrained_embeddings is None:
        pretrained_embeddings = 'emb' in filename
    model = model = EncoderDecoder(
        source_vocab_size=len(word_field.vocab),
        target_vocab_size=len(word_field.vocab),
        save_probs=True,
        d_model= d_model,
        pretrained_embeddings=pretrained_embeddings,
        word_field=word_field).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()

    return model
