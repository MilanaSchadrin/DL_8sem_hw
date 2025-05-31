import torch
from stuff import BOS_TOKEN, EOS_TOKEN, make_mask, tokens_to_words, DEVICE, load_model, words_to_tokens
from data import load_word_field, load_dataset, iterr


def get_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            clean_line = line.strip()
            if clean_line:
                data.append(clean_line)
    return data


@torch.no_grad()
@torch.no_grad()
def summarize(model, source_input_tokens, word_field, beam_width=5, max_len=64, min_words=10, max_words=30):
    model.eval()
    BOS = word_field.vocab.stoi[BOS_TOKEN]
    EOS = word_field.vocab.stoi[EOS_TOKEN]
    PAD = word_field.vocab.stoi['<pad>']

    sequences = [(torch.tensor([[BOS]], device=DEVICE), 0.0)]
    
    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[0, -1].item() == EOS:
                all_candidates.append((seq, score))
                continue

            source_mask, target_mask = make_mask(source_input_tokens, seq, pad_idx=PAD)

            logits = model(source_input_tokens, seq, source_mask, target_mask)
            probs = torch.softmax(logits[:, -1], dim=-1)
            
            topk_probs, topk_idx = probs.topk(beam_width)
            
            for i in range(beam_width):
                token = topk_idx[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, token], dim=1)
                new_score = score + torch.log(topk_probs[0, i]).item()
                all_candidates.append((new_seq, new_score))

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        if all(seq[0, -1].item() == EOS for seq, _ in sequences):
            break

    best_seq = sequences[0][0][0].tolist()
    
    decoded = []
    for tok in best_seq:
        word = word_field.vocab.itos[tok]
        if word not in [BOS_TOKEN, EOS_TOKEN, '<pad>']:
            decoded.append(word)
        if word == EOS_TOKEN:
            break
    
    result = ' '.join(decoded[:max_words])
    
    if len(decoded) < min_words:
        result = ' '.join(decoded + ['...'])
    
    return result.capitalize()


def generate_summaries(output_path, dataset_examples, model, vocab_field):
    with open(output_path, "w", encoding="utf-8") as out_file:
        for example in dataset_examples:
            input_tokens = tokens_to_words(vocab_field, example)
            clean_input = [token for token in input_tokens if token != '<pad>']
            print(f"Input:\n{' '.join(clean_input)}", file=out_file)

            source_tensor = example.view(1, -1).to(DEVICE)
            summary_text = summarize(model, source_tensor, vocab_field)
            print(f"Output:\n{summary_text}\n", file=out_file)


def main(model_path, file_path):
    word_field = load_word_field('./saved_data/wordfield')
    model = load_model(word_field, model_path).to(DEVICE)

    train_dataset = load_dataset('./saved_data/train')
    test_dataset = load_dataset('./saved_data/test')
    _, test_iter = iterr(train_dataset, test_dataset)

    test_examples = next(iter(test_iter)).source.T[:5]
    generate_summaries("results/news_test_summarizations.txt", test_examples, model, word_field)

    examples_from_file = get_data(file_path)
    tokenized = [words_to_tokens(word_field, inp).to(DEVICE).view(1, -1) for inp in examples_from_file]

    with open("results/file_test_sum.txt", "w", encoding="utf-8") as out_file:
        for inp_text, inp_tensor in zip(examples_from_file, tokenized):
            print(f"Input:\n{inp_text}", file=out_file)
            summary_text = summarize(model, inp_tensor, word_field)
            print(f"Output:\n{summary_text}\n", file=out_file)


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main("results/model.pt", "examples/example.txt")
