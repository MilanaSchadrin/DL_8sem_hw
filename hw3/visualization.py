import matplotlib.pyplot as plt
import seaborn as sns
import stuff
from model import EncoderDecoder

#blocks_count=4, heads_count=8

def draw(data, x, y):
    sns.heatmap(data, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, cbar=False)

def visualization_attention(model, word_field, elem):
    source_input, target_input, source_mask, target_mask_ref = stuff.convert_bacth(elem)
    encoder_output = model.encoder(source_input, source_mask)
    words = stuff.tokens_to_words(word_field, elem.source)
    #Encoder
    for layer in range (4):
        for head in range(8):
            draw(model.encoder._blocks[layer]._self_attn._attn_probs[0,head].data.cpu(), words, words)
            plt.savefig(f"..datasets/encoder/encoder_layer_{layer+1}/attn_head_{head+1}.jpg")
    #Decoder