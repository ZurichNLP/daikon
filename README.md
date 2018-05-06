# daikon

A simple encoder-decoder model based on recurrent neural networks (RNNs) for machine translation. Supports model training and translation with trained models.

`daikon` is derived from [`romanesco`](https://github.com/laeubli/romanesco) written by Samuel Läubli.

## Installation

Make sure you have an NVIDIA GPU at your disposal, with all drivers and CUDA
installed. Make sure you also have `python >= 3.5`, `pip` and `git` installed,
and run

```bash
git clone https://github.com/ZurichNLP/daikon
cd daikon
pip install --user -e .
```

If you have sudo privileges and prefer to install `daikon` for all users on
your system, omit the `--user` flag. The `-e` flag installs the app in “editable
mode”, meaning you can change source files (such as `daikon/constants.py`) at any
time.

## Model training

Models are trained from plaintext files with one sentence per line.
Symbols – e.g., words or characters – are delimited by blanks. You need to specify two
parallel files: one in the source language, and one in the target language.

Example file (word-level):

```
I love the people of Iowa .
So that &apos;s the way it is .
Very simple .
```

Example file (character-level):

```
I <blank> l o v e <blank> t h e <blank> p e o p l e <blank> o f <blank> I o w a .
S o <blank> t h a t &apos; s <blank> t h e <blank> w a y <blank> i t <blank> i s .
V e r y <blank> s i m p l e .
```

`daikon` doesn't preprocess training data. If you want to train a model on lowercased input, for example, you'll need to lowercase the training data yourself.

To train a model from `source.txt` and `target.txt` using GPU 0, run

```bash
CUDA_VISIBLE_DEVICES=0 daikon train --source source.txt --target target.txt
```

By default, the trained model and vocabulary will be stored in a directory called `model`, and logs (for monitoring with Tensorboard) in `logs`. You can use custom destinations through the `-m` and `-l` command line arguments, respectively. Folders will be created if they don't exist.

Some hyperparameters can be adjusted from the command line; run `daikon train -h` for details. Other hyperparameters are currently hardcoded in `daikon/constants.py`.


## Translation

A trained model can be used to translate new text. To translate a string on GPU 0 run

```bash
CUDA_VISIBLE_DEVICES=0 echo "Here is a sample input text" | daikon translate
```

This assumes there is a folder called `model` in your current working directory, containing a model trained with `daikon` (see above). If your model is stored somewhere else, use the `-m` command line argument.

For further options, run `daikon translate -h`.


## Scoring

Finally, `daikon` can score existing translations (pairs of source and target sentences):

```bash
CUDA_VISIBLE_DEVICES=0 daikon score --source source.txt --target target.txt 
```

Assuming, again, that there is a folder called `model` in your current working directory that contains a trained model.

For further options, run `daikon score -h`.
