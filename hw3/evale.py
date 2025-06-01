from model import EncoderDecoder
import torch
from stuff import DEVICE, tokens_to_words, BOS_TOKEN, EOS_TOKEN, subsequent_mask, load_model,decoding_tokens
from tqdm.auto import tqdm
import json
import evaluate
from data import load_word_field, load_dataset, iterr
import os
from itertools import islice
import argparse

PAD_TOKEN = '<pad>'

@torch.no_grad()
def batch_generate(model, src, src_mask, word_field, max_len=30):
    batch_size = src.size(0)
    trg_seq = torch.full((batch_size, 1), word_field.vocab.stoi[BOS_TOKEN], device=DEVICE, dtype=torch.long)
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    
    for _ in range(max_len):
        trg_mask = subsequent_mask(trg_seq.size(1)).type_as(src_mask)
        out = model(src, trg_seq, src_mask, trg_mask)  # [batch, len, vocab]
        next_token = out[:, -1, :].argmax(-1, keepdim=True)  # [batch, 1]

        trg_seq = torch.cat([trg_seq, next_token], dim=1)
        finished |= (next_token.squeeze(1) == word_field.vocab.stoi[EOS_TOKEN])
        if finished.all():
            break

    return trg_seq

def eval(model, word_field, data_iter, max_len=20):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(data_iter, desc="Evaluating"):
            src = batch.source.to(DEVICE).transpose(0, 1)  # [batch, seq_len]
            trg = batch.target.to(DEVICE).transpose(0, 1)  # [batch, seq_len]

            for t in trg:
                words = [word_field.vocab.itos[token.item()] for token in t]
                clean_words = [w for w in words if w not in (BOS_TOKEN, EOS_TOKEN, '<pad>')]
                references.append(' '.join(clean_words))

            for i in range(src.size(0)):
                src_seq = src[i].unsqueeze(0)
                src_mask = (src_seq != word_field.vocab.stoi[word_field.pad_token]).unsqueeze(-2)

                trg_seq = torch.tensor([word_field.vocab.stoi[BOS_TOKEN]], device=DEVICE).unsqueeze(0)

                for _ in range(max_len):
                    trg_mask = subsequent_mask(trg_seq.size(-1)).type_as(src_mask)
                    output = model(src_seq, trg_seq, src_mask, trg_mask)

                    pred_token = output.argmax(-1)[:, -1].unsqueeze(-1)
                    trg_seq = torch.cat([trg_seq, pred_token], dim=-1)

                    if pred_token.item() == word_field.vocab.stoi[EOS_TOKEN]:
                        break

                words = [word_field.vocab.itos[token.item()] for token in trg_seq.squeeze(0)]
                clean_words = [w for w in words if w not in (BOS_TOKEN, EOS_TOKEN, '<pad>')]
                pred_text = ' '.join(clean_words)
                predictions.append(pred_text)

    rouge = evaluate.load('rouge')
    results = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    metrics = {
    'rouge1': results['rouge1'],
    'rouge2': results['rouge2'],
    'rougeL': results['rougeL']}

    os.makedirs("metrics", exist_ok=True)

    with open("metrics/rouge_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nROUGE Results:")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="results/model_not_label_sm.pt", help='Path to the model file')
    args = parser.parse_args()
    word_field = load_word_field('./saved_data/wordfield')
    model = load_model(word_field, args.model_path).to(DEVICE)
    train_dataset = load_dataset('./saved_data/train')
    test_dataset = load_dataset('./saved_data/test')
    _, test_iter = iterr(train_dataset, test_dataset)
    eval(model, word_field, test_iter)