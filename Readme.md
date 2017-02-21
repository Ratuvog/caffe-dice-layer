# CUDA implementation [Dice metric](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) for framework Caffe

### How to use
1. Copy repo files in Caffe sources
2. Rebuild caffe
3. Create layer in your model, for example:

`layer {
  name: "dice"
  type: "Dice"
  bottom: "predictions"
  bottom: "groundtruth"
  top: "dice"
}
`

### Features
1. Forward pass layer used for comparing the similarity of two samples.
2. It supports multiple-GPU or CPU.

### Roadmap
 - Backward propogation
 - Using as loss layer
