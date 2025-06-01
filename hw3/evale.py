from model import EncoderDecoder
import torch
from stuff import DEVICE, tokens_to_words, BOS_TOKEN, EOS_TOKEN, subsequent_mask, load_model
from tqdm.auto import tqdm
import json
import evaluate
from data import load_word_field, load_dataset, iterr
import os

def eval(model, word_field, data_iter, max_len=50):
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad(): 
        for batch in tqdm(data_iter, desc="Evaluating"):
            src = batch.source.to(DEVICE)
            trg = batch.target.to(DEVICE)
            
            ref_texts = []
            for t in trg:
                words = tokens_to_words(word_field, t)

                clean_words = [w for w in words if w not in [BOS_TOKEN, EOS_TOKEN, '<pad>']]
                ref_texts.append(' '.join(clean_words))
            references.extend(ref_texts)
            

            for i in range(src.size(0)):  
                src_seq = src[i].unsqueeze(0)
                src_mask = (src_seq != word_field.vocab.stoi[word_field.pad_token]).unsqueeze(-2)
                
                trg_seq = torch.tensor(
                    [word_field.vocab.stoi[BOS_TOKEN]], 
                    device=DEVICE
                ).unsqueeze(0)
                
                for _ in range(max_len):
                    trg_mask = subsequent_mask(trg_seq.size(-1)).type_as(src_mask)
                    output = model(src_seq, trg_seq, src_mask, trg_mask)
                    
                    pred_token = output.argmax(-1)[:, -1].unsqueeze(-1)
                    trg_seq = torch.cat([trg_seq, pred_token], dim=-1)
                    
                    if pred_token.item() == word_field.vocab.stoi[EOS_TOKEN]:
                        break
                
                words = tokens_to_words(word_field, trg_seq.squeeze(0))
                clean_words = [w for w in words if w not in [BOS_TOKEN, EOS_TOKEN, '<pad>']]
                pred_text = ' '.join(clean_words)
                predictions.append(pred_text)

    assert len(predictions) == len(references), \
        f"Mismatch in predictions ({len(predictions)}) and references ({len(references)})"

    rouge = evaluate.load('rouge')
    results = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    
    metrics = {
        'rouge1': results['rouge1'].mid.fmeasure,
        'rouge2': results['rouge2'].mid.fmeasure,
        'rougeL': results['rougeL'].mid.fmeasure
    }
    
    os.makedirs("metrics", exist_ok=True)
    
    with open("metrics/rouge_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nROUGE Results:")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")


if __name__ == "__main__":
    word_field = load_word_field('./saved_data/wordfield')
    model = load_model(word_field, 'results/model.pt').to(DEVICE)
    train_dataset = load_dataset('./saved_data/train')
    test_dataset = load_dataset('./saved_data/test')
    _, test_iter = iterr(train_dataset, test_dataset)
    eval(model, word_field, test_iter)