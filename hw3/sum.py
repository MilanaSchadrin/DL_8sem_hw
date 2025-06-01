import torch
from stuff import BOS_TOKEN, EOS_TOKEN, make_mask, tokens_to_words, DEVICE, load_model, words_to_tokens,subsequent_mask
from data import load_word_field, load_dataset, iterr
import heapq
import torch.nn.functional as F
import html


def get_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            clean_line = line.strip()
            if clean_line:
                data.append(clean_line)
    return data

@torch.no_grad()
def encode_input(model, text, field, device):
    tokens = [field.init_token] + field.tokenize(text) + [field.eos_token]
    indexed = [
        field.vocab.stoi.get(tok, field.vocab.stoi[field.unk_token])
        for tok in tokens
    ]
    src_tensor = torch.LongTensor(indexed).unsqueeze(0).to(device)
    src_mask = (src_tensor != field.vocab.stoi[field.pad_token]).unsqueeze(1).unsqueeze(2)
    encoder_outputs = model.encoder(src_tensor, src_mask)
    return encoder_outputs, src_mask


def beam_search_decode(model, encoder_outputs, src_mask, field, max_len, beam_size, len_penalty, ngram_block, device):
    init_id = field.vocab.stoi[field.init_token]
    eos_id = field.vocab.stoi[field.eos_token]
    pad_id = field.vocab.stoi[field.pad_token]

    beams = [(0.0, [init_id])]

    for step in range(max_len):
        all_candidates = []

        for score, seq in beams:
            trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            trg_mask = (trg_tensor != pad_id).unsqueeze(1).unsqueeze(2)
            trg_mask = trg_mask & subsequent_mask(trg_tensor.size(1)).to(device)

            logits = model.decoder(trg_tensor, encoder_outputs, src_mask, trg_mask)
            log_probs = F.log_softmax(logits[:, -1], dim=-1)

            # N-gram blocking
            if ngram_block > 0 and len(seq) >= ngram_block:
                prefix = tuple(seq[-(ngram_block - 1):])
                blocked = set()
                for i in range(len(seq) - ngram_block + 1):
                    if tuple(seq[i:i + ngram_block - 1]) == prefix:
                        blocked.add(seq[i + ngram_block - 1])
                for tok_id in blocked:
                    log_probs[0, tok_id] = -1e9
            if step < 4:
                log_probs[0, eos_id] -= 1.0

            top_probs, top_indices = log_probs.topk(beam_size)

            for prob, idx in zip(top_probs[0], top_indices[0]):
                new_seq = seq + [idx.item()]
                new_score = score + prob.item()
                all_candidates.append((new_score, new_seq))

        beams = heapq.nlargest(
            beam_size,
            all_candidates,
            key=lambda x: x[0] / ((len(x[1]) ** len_penalty) if len_penalty > 0 else 1)
        )

        if all(seq[-1] == eos_id for _, seq in beams):
            break

    return max(beams, key=lambda x: x[0])[1]


def finalize_summary(token_ids, field):
    unwanted = {'<pad>', '<unk>', '<sos>', '<eos>', BOS_TOKEN, EOS_TOKEN}
    tokens = [field.vocab.itos[i] for i in token_ids]
    tokens = [html.unescape(tok) for tok in tokens if tok not in unwanted]
    tokens = [tok for tok in tokens if len(tok) > 1 or tok.isalpha()]
    return ' '.join(tokens).strip().capitalize()


def summarize(
    model, input_text, word_field,
    max_summary_len=25, beam_size=8, len_penalty=0.9, block_ngram=3,
    device=DEVICE
):
    model.eval()

    encoder_outputs, src_mask = encode_input(model, input_text, word_field, device)

    best_sequence = beam_search_decode(
        model, encoder_outputs, src_mask, word_field,
        max_len=max_summary_len, beam_size=beam_size,
        len_penalty=len_penalty, ngram_block=block_ngram,
        device=device
    )

    return finalize_summary(best_sequence[1:-1], word_field) 


def generate_summaries(output_path, dataset_examples, model, vocab_field):
    with open(output_path, "w", encoding="utf-8") as out_file:
        for example in dataset_examples:
            input_tokens = tokens_to_words(vocab_field, example)
            clean_input = [token for token in input_tokens if token != '<pad>']
            input_text = ' '.join(clean_input)
            print(f"Input:\n{' '.join(clean_input)}", file=out_file)

            summary_text = summarize(model, input_text, vocab_field)
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
            summary_text = summarize(model, inp_text, word_field)
            print(f"Output:\n{summary_text}\n", file=out_file)


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main("results/model.pt", "examples/example.txt")
