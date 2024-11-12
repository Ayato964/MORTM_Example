import matplotlib.pyplot as plt
from pretty_midi import  PrettyMIDI, Instrument, Note

def add_tempo_notes(midi: PrettyMIDI, save_directory:str):
    inst: Instrument = Instrument(program=10)
    met = 0
    count = 0
    tempo = midi.get_tempo_changes()[-1]
    while midi.get_end_time() > met:
        inst.notes.append(Note(pitch=80 if not count == 0 else 92, velocity=100, start=float(met), end=0.5))
        met += 60 / tempo
        count += 1
        if count == 4:
            count = 0
    midi.instruments.append(inst)
    midi.write(save_directory)


midi = PrettyMIDI("out/generate_test.midi")

add_tempo_notes(midi, "out/generate_test_add_metro.midi")

piano_roll = midi.get_piano_roll(fs=100)  # fsは1秒あたりのフレーム数（解像度）

# プロット
plt.figure(figsize=(10, 6))
plt.imshow(piano_roll, aspect='auto', cmap='coolwarm', origin='lower')
plt.xlabel('Time (frames)')
plt.ylabel('MIDI Note')
plt.title('Piano Roll')
plt.colorbar(label='Velocity')
plt.show()