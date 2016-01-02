# rnn summary

My own notes on how rnn works.  Mostly derivatives from the rnn documentation, but supplemented a bit by reading the source code.

Emphasis is on finding out enough to get char-rnn-er working :-)  Everything else is out of scope really

## Outstanding issues in char-rnn-er

- Training pairs of input/output values using normal non-rnn training works ok
- But training multiple pairs at a time, using backwardsonline, fails to train the last pair of each sequence for some reason

## What happens in absence of rnn?

- we have a network, in this case an nn.Sequential
  - contains an nn.Linear and an nn.LogSoftMax
- we train on each pair by doing:
  - net:forward(input)
  - (get gradOutput from criterion; criterion usage doesnt change between backwardsonline or non-rnn usage, so we assume it works correctly)
  - net:backward(input, gradOutput)
  - net:updateParameters(learningRate)
- net:forward will call 

## Class hierarchies

```
Sequential => Container  => Module
Linear                   => Module
LogSoftMax               => Module

Sequential:
  updateOutput => modules:updateOutput
  updateGradInput => modules:updateGradInput
  accUpdateGradParameters => modules:accUpdateGradParameters
  accGradparameters => modules:accGradParameters
  backward => modules:backward

Container:
  zeroGradParameters modules:zeroGradParameters
  updateParameters modules:updateParameters
  parameters concatenate modules:parameters
  training modules:training
  evaluate modules:evaluate
  applyToModules

Module:
  updateOutput return self.output
  updateGradInput return self.gradInput
  accGradParameters nothing
  accUpdateGradParameters self:accGradParameters
  forward  self:updateOutput
  backward self:updateGradInput, self:accGradParameters
  backwardUpdate  self:updateGradInput, self:accUpdateGradParameters
  zeroGradParameters zeros parameters()
  updateParameters  adds learningrate * self.parameters()[2] to parameters()[1]
  training self.train = true
  evaluate self.train = false
  clone   clone, via serialize to memory file
  flatten  flattens all parameters from self and children into single storage
  getParameters

Linear:
  updateOutput  calc output
  updateGradInput  calc gradInput
  addGradParameters  calc gradWeight, gradBias

LogSoftMax:
  updateOutput
  updateGradInput 
```
