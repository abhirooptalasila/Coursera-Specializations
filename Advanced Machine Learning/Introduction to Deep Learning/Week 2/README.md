# Introduction to Deep Learning

- [Introduction to Deep Learning](#introduction-to-deep-learning)
  - [Multilayer perceptron (MLP)](#multilayer-perceptron-mlp)
    - [Chain Rule](#chain-rule)
    - [Backpropagation](#backpropagation)
  - [Matrix Derivatives](#matrix-derivatives)
    

---

## Multilayer perceptron (MLP)

The following slide summarises the linear binary classifier.

<img src="../../images/binlr.png" width="400"/>

Another similar task algorithm is Logistic Regression. We can think of it like calculating the distance of a point from the decision function. We convert that distance into probabilities in the range [0,1] using a Sigmoid function. 

<img src="../../images/cg.png" width="400"/>

This is a basic neural network consisting of multiple neurons.


### Chain Rule

We can use chain rule to compute derivates of composite functions. We can use a computation graph of derivates to compute them automatically.

<img src="../../images/cr.png" width="400"/>

A really intuitive way to visualize a derivatives graph: 

<img src="../../images/dr.png" width="400"/>
<img src="../../images/cr1.png" width="400"/>


### Backpropagation

<img src="../../images/brp.png" width="400"/>

This is called reverse-mode differentiation.
In application to neural networks it has one more name: back-propagation. It works fast, because we reuse computations from previous steps. 

<img src="../../images/bpi.png" width="400"/>

---

## Matrix Derivatives

<img src="../../images/mmul.png" width="400"/>

<img src="../../images/mlpm.png" width="400"/>

<img src="../../images/fpi.png" width="400"/>
