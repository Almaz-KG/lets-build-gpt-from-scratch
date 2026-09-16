# lets-build-gpt-from-scratch

My own implementation of the GPT that Andrej Karpathy builds in [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).

Typed out by hand while following the video, not cloned from [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture).
The reference repository exists and I read it, but only after my own version of a stage runs.

## Why

I have read the transformer paper, the annotated versions of it, and a fair amount of code that uses one.
None of that tells me whether I can write one.

The gap is specific and it is not conceptual.
I know what attention does.
What I do not know, without checking, is which axis the softmax goes over, why the mask is `-inf` before the softmax rather than a zero after it, where the `/sqrt(d_k)` actually matters, and what `x = x + self.sa(self.ln1(x))` does to the gradient that `x = self.ln1(x + self.sa(x))` does not.
Those only come from writing it, breaking it, and watching the loss refuse to move.

So the goal is not to end up with a GPT.
A checkpoint is worth nothing here: the model is character-level, trained on one megabyte of Shakespeare, and its output is Shakespeare-shaped nonsense.
The goal is to end up having written one, and to have the shapes come out of the fingers afterwards.

This is the same bet as [pytorch-exercises-for-llm-developers](../pytorch-exercises-for-llm-developers), one level up.
There the drills are isolated and synthetic.
Here they have to hold together as a single model that trains.

## What gets built

A decoder-only transformer, character-level, on Tiny Shakespeare.

The end state matches the video: around 10M parameters, six layers, six heads, embedding width 384, context 256 characters, dropout 0.2.
That is small enough to train on a single GPU in minutes and large enough that every architectural piece has to be there for the loss to come down.

The video stops one step short of a language model anyone would use, and says so.
Pretraining is the whole of what is built here.
Everything that turns a pretrained model into an assistant, the supervised fine-tuning and the reward model and the RL on top, is named at the end of the video and not implemented in it.
It is not implemented here either.

## The stages

The bigram baseline is written and runs.
Everything after it is the shape the work is heading towards, and the stages land one at a time.

Each stage is its own runnable file that trains and samples on its own, rather than a single model that grows through git history.
That is deliberate: the interesting comparison is between two stages, so both have to still exist.

```
data/input.txt      Tiny Shakespeare, about 1.1 MB, the only dataset
01_bigram.py        the baseline: an embedding table, no attention at all
02_single_head.py   one head of self-attention, the masked-softmax core
03_multi_head.py    heads in parallel, plus the feed-forward
04_blocks.py        blocks, residual connections, layer norm
05_gpt.py           scaled up and regularised, the model from the end of the video
notes.md            what broke, and what the fix turned out to be
```

The bigram baseline is the one that matters most and the one easiest to skip.
It fixes the loss that everything after it has to beat, and it is the only stage where the number the model is going to reach can be worked out on paper first.

`notes.md` is the actual output of this repository.
The code is going to converge on Karpathy's within a few characters, because it is his architecture.
What will not be in his repository is the list of things I got wrong on the way there.

## Running

A standalone `uv` project, the same as every other subproject in this workspace:

```sh
uv sync                      # once
uv run python 01_bigram.py   # train a stage and sample from it
```

Run from the repository root, since the dataset path is relative.

`data/input.txt` is not committed, and nothing fetches it automatically yet, so a fresh clone needs it first:

```sh
mkdir -p data && curl -o data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Stage 01 takes about three seconds on cpu and reports a loss of 4.72 before training, then 2.53 on train and 2.55 on validation after 10 000 steps.

## What this is not

Not [nanoGPT](https://github.com/karpathy/nanoGPT), which is the maintained, scaled, reproduce-GPT-2 version of the same idea.
Not a training framework, not a library, and not something to depend on.
It is one person working through a two-hour video in the only format that proves the thing landed.
