from mortm.de_convert import ct_token_to_midi
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC
import numpy as np
import torch


tokenizer = Tokenizer(get_token_converter( TO_MUSIC))
tokenizer.rev_mode()
output = np.load("./ex/Sample4.mid.npz")['array1']

output = torch.tensor(output.tolist())

print(output)
midi = ct_token_to_midi(tokenizer, output, "out/generate_test.midi", program=65)
