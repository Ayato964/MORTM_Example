# MORTM 1.0 Beta-5 Release

## Release Note

### 2024/10/17
* New tokenizer and reduced noise.
* Expression increased.

### Hyper Parameter
d_model: 1024

trans_layer: 12

d_ff: 4096

batch_size: 32

lr : None

loss(min): 2.4

# About
This model can generate Melodies.

Generating a continuation from an existing piece of music is more accurate than generating a melody from scratch.

You may use by marging this repository or open the following URL and execute Google colabo. 

executable procedure is the following.

## How to use

### Use marge

1. Marge this repository.
2. Open the this [Google Drive](https://drive.google.com/drive/folders/1vOanIV1Po09KRZfMFglWdKjACIr1chtZ?usp=sharing) and download MORTM model and vocab_list.json.
3. open the [convert.py](convert.py) module and run.

   You need to rewrite "Sample.mid" if convert your MIDI data.
4. If complete execute, save NPZ file in the [./ex](./ex/) directory.
5. Open the main.py module and run.

   If change sample midi, you need to rewrite np.load method's argument.

   ```np.load("Sample.npz")```
6. Listen to generatble MIDI file.

### Use Colabo


