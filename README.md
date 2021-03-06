# Provably robust neural networks

*A repository for training provably robust neural networks by optimizing
convex outer bounds on the adversarial polytope. Created by [Eric Wong](https://riceric22.github.io) and [Zico Kolter](http://zicokolter.com). [Link to arXiv paper][paper].*

[paper]: https://arxiv.org/abs/1711.00851

## News
+ 4/26/2018 - Added robust models from the paper to the `models/` folder in the
repository. 
+ 3/4/2018 - Updated paper with more experiments. Code migrated to work with
  PyTorch 0.3.0+. Real mini-batching implemented, with a 5x speedup over the
  old codebase, and several NaN bugfixes. 
+ 12/8/2017 - Best defense paper at the NIPS 2017 ML & Security Workshop
+ 11/2/2017 - Initial preprint and release of codebase. 

## Installation & Usage
You can install this repository with 
`pip install convex_adversarial`. The package contains the following functions: 
+ `robust_loss(net, epsilon, X, y, 
                size_average=True, alpha_grad=False, scatter_grad=False)`
    computes a robust loss function for a given ReLU network `net` and l1 
    radius `epsilon` for examples `X` and their labels `y`. You can use 
    this as a drop in replacement for, say, `nn.CrossEntropyLoss`, and is
    equivalent to the objective of Equation 14 in the paper. 
+ `DualNetBounds(net, X, epsilon, alpha_grad=False, scatter_grad=False)`
    is a class that computes the layer-wise upper and lower bounds for all
    activations in the network. This is useful if you are only interested 
    in the bounds and not the robust loss, and corresponds to Algorithm 
    1 in the paper. 
+ `DualNetBounds.g(self, c)` is a class function that computes the lower
    bound on the primal problem described in the paper for a given 
    objective vector c. This corresponds to computing objective of Theorem 1 in
    the paper (Equation 5). 

## Why do we need robust networks? 
While networks are capable of representing highly complex functions. For
example, with today's networks it is an easy task to achieve 99% accuracy on
the MNIST digit recognition dataset, and we can quickly train a small network
that can accurately predict that the following image is a 7.

<img src="https://github.com/locuslab/convex_adversarial.release/blob/master/images/seven.png" width="100">

However, the versatility of neural networks comes at a cost: these networks
are highly susceptible to small perturbations, or adversarial attacks (e.g. the [fast gradient sign method](https://arxiv.org/abs/1412.6572) and [projected gradient descent](https://arxiv.org/abs/1706.06083))! While
most of us can recognize that the following image is still a 7, the same
network that could correctly classify the above image instead classifies 
the following image as a 3.

<img src="https://github.com/locuslab/convex_adversarial.release/blob/master/images/seven_adversarial.png" width="100">

While this is a relatively harmless example, one can easily think of
situations where such adversarial perturbations can be dangerous and costly
(e.g. autonomous driving). 

## What are robust networks? 
Robust networks are networks that are trained to protect against any sort of
adversarial perturbation. Specifically, for any seen training example, the
network is robust if it is impossible to cause the network to incorrectly
classify the example by adding a small perturbation.

## How do we do this? 
The short version: we use the dual of a convex relaxation of the network over
the adversarial polytope to lower bound the output. This lower bound can be
expressed as another deep network with the same model parameters, and
optimizing this lower bound allows us to guarantee robustness of the network.

The long version: see our paper, [Provable defenses against adversarial examples via the convex outer adversarial polytope][paper]. 

## What difference does this make? 
We illustrate the power of training robust networks in the following two scenarios: 2D toy case for a visualization, and on the MNIST dataset. More experiments are in the paper. 

### 2D toy example
To illustrate the difference, consider a binary classification task on 2D
space, separating red dots from blue dots. Optimizing a neural network in the
usual fashion gives us the following classifier on the left, and our robust
method gives the classifier on the right. The squares around each example
represent the adversarial region of perturbations.

<p float="left">
<img src="https://github.com/locuslab/convex_adversarial.release/blob/master/images/normal_trained.png" width="300">
<img src="https://github.com/locuslab/convex_adversarial.release/blob/master/images/robust_trained.png" width="300">
</p>

For the standard classifier, a number of the examples have perturbation
regions that contain both red and blue. These examples are susceptible to
adversarial attacks that will flip the output of the neural network. On the
other hand, the robust network has all perturbation regions fully contained in
the either red or blue, and so this network is robust: we are guaranteed that
there is no possible adversarial perturbation to flip the label of any
example.

### Robustness to adversarial attacks: MNIST classification
As mentioned before, it is easy to fool networks trained on the MNIST dataset 
when using attacks such as the fast gradient sign method (FGS) and projected gradient descent (PGD). We observe that PGD can almost always fool the MNIST trained network. 

|          | Base error | FGS error | PGD Error | Robust Error |
| --------:| ----------:|----------:| ---------:| ------------:|
| Original |       1.1% |     50.0% |     81.7% |         100% |
|   Robust |       1.8% |      3.9% |      4.1% |         5.8% |

On the other hand, the robust network is significantly less affected by these
attacks. In fact, when optimizing the robust loss, we can additionally
calculate a *robust error* which gives an provable upper bound on the error
caused by *any* adversarial perturbation. In this case, the robust network has
a robust error of 5.8%, and so we are guaranteed that no adversarial attack
can ever get an error rate of larger than 5.8%. In comparison, the robust
error of the standard network is 100%. More results on HAR, Fashion-MNIST, and
SVHN can be found in the [paper][paper]. 

## What is in this repository? 
+ The code implementing the robust loss function that measures the convex
  outer bounds on the adversarial polytope as described in the paper. It is
  implemented for linear and convolutional networks with ReLU activation on all
  layers except the last. 
+ Examples, containing the following: 
  + Code to train a robust classifier for the MNIST, Fashion-MNIST, HAR, and SVHN datasets. 
  + Code to generate and plot the 2D toy example.
  + Code to find minimum distances to the decision boundary of the neural network
  + Code to attack models using FGS and PGD
  + Code to solve the primal problem exactly using CVXPY
