from mortm.de_convert import ct_tokens_to_midi
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC
import numpy as np
import torch


tokenizer = Tokenizer(get_token_converter(120, TO_MUSIC), "./model/vocab/vocab_list.json")

output = np.load("./ex/Sample.mid.npz")['array1']

output = torch.tensor(output.tolist())

print(output)
midi = ct_tokens_to_midi(tokenizer, output, "out/generate_test.midi")
