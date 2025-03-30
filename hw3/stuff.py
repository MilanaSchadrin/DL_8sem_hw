import torch

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
    DEVICE = torch.device('cuda')
else:
    from torch import FloatTensor, LongTensor
    DEVICE = torch.device('cpu')

def tokens_to_words(word_filed, token_tensor):
    tokens = token_tensor.view(-1).tolist()
    return [word_filed.vocab.itos[token] for token in tokens]

def words_to_tokens(word_field, word_list):
    tokens = [word_field.vocab.stoi.get(word, word_field.vocab.stoi['<unk>']) 
              for word in word_list]
    tokens = [word_field.vocab.stoi[BOS_TOKEN]] + tokens + [word_field.vocab.stoi[EOS_TOKEN]]
    return torch.tensor(tokens).unsqueeze(1) 

def subsequent_mask(size):
    mask = torch.ones(size, size, device=DEVICE).triu_()
    return mask.unsqueeze(0) == 0

def make_mask(source_inputs, target_inputs, pad_idx):
    source_mask = (source_inputs != pad_idx).unsqueeze(-2)
    target_mask = (target_inputs != pad_idx).unsqueeze(-2)
    target_mask = target_mask & subsequent_mask(target_inputs.size(-1)).type_as(target_mask)
    return source_mask, target_mask


def convert_batch(batch, pad_idx=1):
    source_inputs, target_inputs = batch.source.transpose(0, 1), batch.target.transpose(0, 1)
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx)

    return source_inputs, target_inputs, source_mask, target_mask