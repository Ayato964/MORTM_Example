from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC, TO_TOKEN
from mortm.convert import MIDI2Seq
import os


TARGET_PROGRAM_NUMBER = [0]

tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

count = 0
con = MIDI2Seq(tokenizer, "./ex", "Sample4.mid", TARGET_PROGRAM_NUMBER)
con.convert()
print(con.aya_node)
is_saved, reason = con.save("./ex/")
print(is_saved, reason)

#tokenizer.save("model/vocab/")
