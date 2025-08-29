### This is an attempt to program a neural net from scratch in C, to test my understanding of deep learning fundamentals.

This repo contains a `.c` implementation of a flexible, fully connected feed-forward neural net, whose functions are accessible via `network.h`.  
Layer dimensions, count, and other hyperparameters are adaptable by changing the macros.  

Im using it to classify 28x28 images of numbers. The network has size **784x300x10** and is trained on **MNIST**.  
Im using **softmax + cross-entropy** for the output layer, **LeakyReLU** for the hidden layer, and basic **gradient descent** for backpropagation.

_TODO: refactoring and adding saving and loading funcs_ 

### Future Plans
I plan on further optimizing by using **Adam** and **mini-batches**.

## Usage

To build the project, simply run:

```bash
make
```
to train MNIST 
```bash
./main
```

## License
MIT License
