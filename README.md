# model-damage
## Overview
This repo provides scripts to progressively remove weights and retrain remaining 'healthy' weights in convolutional network to simulate neurodegeneration. 

RDM figure generation in rdm_graphs.py can be used with your own saved activations of any model, the code will just need some adapting 

## Install
```pip install requirements.txt```

## Usage
Load a model into ```prune_retrain_compressed.py```. In this script, we use a compressed version of VGG19 trained on CIFAR10. This model and data are provided in the engine but any CNN model may be used. Hyperparameters such as how many steps of degeneration (num_injury_steps) and the percent of weights to ablate in one step (percent_to_injure) may be adjusted. 

The full model is evaluated and activations from the penultimate layer are stored and sorted into a dictionary according to their class. 

The model weights in Conv2D and Linear layers are then progressively and randomly pruned, with retraining in between each iteration of injury. Each time the model is injured and retrained, activations from the penultimate layer are saved and representational dissimilarity matrices (RDMs) are generated. 

### Visualization
RDM visualizations are generated with ```rdm_graphs.py```. This script generated activations on the test set of images from CIFAR10 but users can input their own pre-saved activations as well. 
