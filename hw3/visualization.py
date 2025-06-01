import matplotlib.pyplot as plt
import seaborn as sns
import os
from sum import summarize
from data import load_word_field
from stuff import DEVICE, load_model_vis, words_to_tokens
import math
import argparse

#blocks_count=4, heads_count=8

def draw(data, x, y):
    sns.heatmap(data, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, cbar=False)

class Visualizer:
    def __init__(self, model, word_field, source_text, generated_summary):
        self.enc_attn = model.encoder.attn_probs
        self.dec_self_attn = model.decoder.self_attn_probs
        self.cross_attn = model.decoder.enc_attn_probs

        tokenize = lambda text: [word_field.init_token] + word_field.tokenize(text.lower()) + [word_field.eos_token]
        self.source_tokens = tokenize(source_text)
        self.summary_tokens = tokenize(generated_summary)

    def plot(self, layer, head, mode='encoder', save_path=None):
        if mode == 'encoder':
            attn = self.enc_attn[layer][0][0]
            attn = attn[head].detach().cpu().numpy()
            x_labels = self.source_tokens
            y_labels = self.source_tokens
        elif mode == 'decoder':
            attn = self.dec_self_attn[layer][0][0]
            attn = attn[head].detach().cpu().numpy()
            x_labels = self.summary_tokens
            y_labels = self.summary_tokens
        elif mode == 'mix':
            attn = self.cross_attn[layer][0][0]
            attn = attn[head].detach().cpu().numpy()
            x_labels = self.source_tokens
            y_labels = self.summary_tokens

        #if attn.ndim == 1:
            #attn = attn.reshape(-1, 1) 

        plt.figure(figsize=(10, 8))
        draw(attn, x_labels, y_labels)
        plt.title(f"{mode.capitalize()} Attention - Layer {layer+1}, Head {head+1}")
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def generate_attention_maps(model, text, word_field, output_root='vis', example_num=1):
    example_dir = os.path.join(output_root, f"ex{example_num}")
    tokens = words_to_tokens(word_field, text).to(DEVICE).view(1, -1)
    summary = summarize(model, tokens, word_field)
    visualizer = Visualizer(model, word_field, text, summary)

    num_layers = len(model.encoder._blocks)
    num_heads = model.encoder._blocks[0]._self_attn._heads_count

    for mode in ['encoder', 'decoder', 'mix']:
        for layer in range(num_layers):
            for head in range(num_heads):
                save_path = os.path.join(example_dir, mode, f"layer_{layer+1}", f"head_{head+1}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                visualizer.plot(layer, head, mode, save_path)

    return summary


def main(model_path):
    word_field = load_word_field('./saved_data/wordfield')
    model = load_model_vis(word_field, model_path).to(DEVICE)
    model.eval()

    texts = [
        "В испанском курортном городе Марбелья сегодня днём произошло серьёзное дорожно-транспортное происшествие. Автомобиль, управляемый 45-летним туристом из Германии, на большой скорости врезался в толпу пешеходов на центральной набережной. По предварительным данным, пострадали не менее 12 человек, трое находятся в критическом состоянии. Местные власти уже начали расследование инцидента.",
        "Крупный пожар произошёл ночью в центре Москвы в многоэтажном офисном здании на улице Тверской. Огонь охватил три верхних этажа, пожарным потребовалось более четырёх часов, чтобы полностью ликвидировать возгорание. К сожалению, в результате происшествия погибли два сотрудника охранной службы, которые находились на дежурстве. Причины пожара устанавливаются.",
        "Компания Google объявила о запуске нового алгоритма поиска, основанного на искусственном интеллекте. Система под кодовым названием 'MUM' способна понимать сложные многоэтапные запросы и контекст вопросов. По заявлению разработчиков, это революционное изменение в поисковых технологиях, которое в корне изменит способ взаимодействия пользователей с интернетом.",
    ]
    i=1
    for i, text in enumerate(texts, start=1):
        print(f"\nInput: {text}")
        summary = generate_attention_maps(model, text, word_field, output_root='vis', example_num=i)
        print(f"Summary: {summary}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization of attention maps")
    parser.add_argument('--model_path', type=str, default="results/model_not_label_sm.pt",
                        help="Path to the trained model checkpoint")
    args = parser.parse_args()
    main(args.model_path)