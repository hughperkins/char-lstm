# char-lstm
char-rnn, tweaked to use Element-Research rnn modules

What this does, and the way it works, is closely based on how Karpathy's https://github.com/karpathy/char-rnn works, but tweaked to use Element Research's https://github.com/element-research/rnn modules instead.

## Status

Draft, not yet fully working

Update:
- both training and sampling are implemented now, but seems to be some critical bug in training for some reason.  I'm working on this :-)

## Differences from original char-rnn

* uses Element Research's rnn modules
* weights are stored as a FloatTensor, rather than CudaTensor etc
* can train using any of cuda/cl/cpu, and sample using the same, or different, up to you
* the sequences used to train each epoch are offset by 1 character from the previous epoch, which hopefully will improve generalization
* each thread is exposed to the entire training set, rather than a 1/batchSize portion of it, which hopefully means can use really large batch sizes, for speed of execution

## Does it support cuda and OpenCL?

* of course it supports OpenCL :-)
* and it supports CUDA :-)

## To do

* currently, calls 'forget' before each sequence.  It should not
* implement sampling
* add command-line options

## How to use

### Training

```
th train.lua
```

### Sampling

eg:
```
th sample.lua out/weights_tinyshakespeare_1_501.t7
```

## Naming

If you can think of a better name, please raise an issue to suggest :-)

## License

Original char-rnn code is MIT.  New code in this repo are BSDv2

