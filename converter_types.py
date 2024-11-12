from mortm.tokenizer import TO_TOKEN, TO_MUSIC, PITCH_TYPE, START_TYPE, DURATION_TYPE
from mortm.aya_node import StartRE, Pitch, Duration, Token
from typing import List

def get_token_converter_mortv2() -> List[Token]:
    register: List[Token] = list()

    #register.append(MeasureToken(MEASURE_TYPE, convert))
    register.append(StartRE( START_TYPE, TO_MUSIC))
    register.append(Pitch( PITCH_TYPE,TO_MUSIC))
    #register.append(Velocity(tempo, VELOCITY_TYPE))
    register.append(Duration(DURATION_TYPE, TO_MUSIC))

    return register