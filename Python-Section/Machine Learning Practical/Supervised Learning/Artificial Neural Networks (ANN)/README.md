### **Artificial Neural Network (ANN)**

Welcome to the section dedicated to Artificial Neural Networks (ANN) within this repository. To gain a deep understanding of ANNs, in this README there is a start with the theoretical foundations. To reinforce the understanding, there are hands-on projects using public datasets. Some practical aspects include: 1. **Implementing ANNs**; 2. **Dataset Exploration**; 3. **Training Models**; 4. **Hyperparameter Tuning**; and 5. **Evaluating Models**

Table of content:

Section 1 - [**Definition of ANN**](#section1)

Section 2 - [**Activation Functions**](#section-2)

Section 3 - [**Gradient Descent**](section-3)

Section 4 - [**Optimizers**](section-4)

Section 5- [**Dropout**](section-5)

Section 6 - [**Hyper parameter tuning in ANN**](section-6)

Section 7 - [**Batch normalization**](section-7)

____
#<a name="section-1">
#### **What is a Artificial Neural Network?**
</a> 
A neural network is a series of algorithms that help to recognize relationships in a dataset through a process which mimics the way human brain works. In this way, it can adapt to Changing Input and generate the Best Results for it.

A neuron in a neural network is a **mathematical function**. It collects and classifies information according to a **specific architecture**.

There are 3 major components in a Neural Network:

![image](https://miro.medium.com/v2/resize:fit:1400/1*f9XlMlruW7TMF3EHbPDfYg.png)

- **Input Layer**: it's composed of artificial input neurons and brings down the initial data into the system for further processing by subsequent layers of aritificial neural networks.
- **Hidden Layer**: one or more and it's located between the input and output of the artificial neural network architecture. The hidden layer applies weights to the inputs and directs them through an activation functions as the output. Hidden layers perform **nonlinear transformations** of the inputs entered into the network.
- **Output Layer**: it produces output for the program.

____

#<a name="section-2">
#### **Activation Functions**
</a>

The hidden layers in artificial neural network use **activation functions to transform the data provided by the input layer**. Activation functions are mathematical equations that determine the output of a neural network. The function is attached to each neuron in the network, and determines wether it should be activated or not, based on wheter each neuron's input is relevant for the **model's prediction**. 

> Activation function: it's a transformation function that maps the input signals into output signals that are needed for the neural network to function.

The activation function helps to determine the output of the program by normalizing the output values in a range of 0 to 1 or -1 to 1.

Types of Activation Function:
- **Linear Activation Function**: the function is a line or linear. Therefore the output of the functions will not be confined between any range. The **equation for this line would be** f(x) = x, and the range would be from -infinity to +infinity. The problem with linear activation functions is that the don't help with the complexity of a problem or various parameters of the usual data that is fed to the neural networks.
- **Non Linear Activation Function**: they are the most used activation functions because the their non-linearity. These nonlinear functions make easy for the model to **generalize or adapt with a variety of data** and to differentiate between the output. The main terminologies needed to understand a nonlinear functions are derivatuve or differential. **Derivative of Differential**: change in y-axis w.r.t. change in x-axis. **Monotonic functions**: a function which is either entirely non-increasing or non-decreasing.

Basic activation functions:
- **Sigmoid Activation Function**: a sigmoid function is a mathematical function having a characteristic 'S'-shaped curve or sigmoid curve. The main reason behind using the sigmoid function is that this function exists between 0 to 1. Therefore, it's especially used for models where it's necessary to perform binary classification. **It can also be used to predict the probability (0 to 1) of an outcome**.

- **Tanh Activation Function**: the range of the tanh function is -1 to +1. Tanh function also looks like an S-shaped curve. **The main difference here is the area of higher slope**. Tahn activation has large area under better slope compared to sigmoid, this helps model using Tanh actiavation to learn better. This activation is generally used for classification between two classes.

Most popular activation functions:
- **ReLU** (reactified linear unit): it's the most widely used and accepted activation function in the world. **The ReLU activation function is generally used for CNN**. It's half rectified from bottom, which means if you feed any negative data to ReLU, it will return zero, and if you feed any positive data, it will return the exact number. The range of a ReLU function is from zero to infinity. **The major issue with a ReLU function is that all the negative values become zero immediately**, which decreases the ability of the model to fit or train from data properly.

- **Leaky ReLU**: it helps to **increase the range of the ReLU function**. It is usually set to a small positive value, such as 0.01, but it can be adjusted based on the problem and dataset.  In the standard ReLU, when the input is less than zero, the gradient of the function becomes zero, leading to neurons that no longer contribute to the learning process. With **Leaky ReLU, even if the input is negative, there is still a small gradient, allowing some learning to occur**.

![image](https://www.researchgate.net/publication/335845675/figure/fig3/AS:804124836765699@1568729709680/Commonly-used-activation-functions-a-Sigmoid-b-Tanh-c-ReLU-and-d-LReLU.ppm)

**When to use Sigmoid and Sofmax activation function**

Sigmoid or Tanh activation function can be used at the final layer to solve binary classification problems. Some cases the problem is a multi-class classification problems, where prediction can be one among n different classes. Sigmoid activation function fails to address this type of problem because of its range being from 0 to 1. The sofmax considers each class (probability weights) in a multi-class classification. Theo output will be the class with the highest probability. Sofmax can also be used for binary classification problems (it will use two neurons in the output layer and sigmoid can solve it with one neuron), but it's recommended to use Softmax while predicting more than two classes to avoid redundancy of the outcome and extra complexity.
- **Sigmoid**: it's suitable for problems where each input can belong to one of several classes. _ It's suitable when you have two mutually exclusive classe._
- **Softmax**: it takes multiple inputs and transforms them into a probability distribution over all the classes, ensuring that the sum of the probabilities for all classes is equal to 1. _It's suitable for problems where each input can belong to one of several classes._

___

#<a name="section-3">
#### **Gradient Descent** (also known as 'back propagation')
</a>

![image](https://sebastianraschka.com/images/faq/gradient-optimization/ball.png)

It's an iterative optimization technique that is used to improve deep learning and neural network based models by minimizing the loss function. In each iteration of gradient descent, the model parameters are updated by a small step (learning rate) in the direction opposite to the gradient of the loss function. 

- **Loss function**: it quantifies the **amount of error in the predictions** (difference between the predicted values (output of a model) and the actual target values (ground truth)). It's a metric that is directly related to the performance of the model. Good: would output a lower number; Bad: higher number. **The goal of a ML model is to learn a set of parameters that minimize the loss function**. To create a own loss function should always justify two things:
    - It should be **negative oriented**, which means lower is always better. 
    - It should be **differentiable**, which mean it should be able to apply derivative on it.
- **Minimization**: it's the process of finding the values of the model's parameters that lead to the lowest possible value of the loss function.

There are two popular types of gradient descent algorithms which are widely used: **Batch Gradient Descent** and **Stochastic Gradiente Descent** algorithm.

##### **Batch vs. Stochastic**

**Batch Gradient Descent:** two main parts are: gradient calculation, which is nothing but slope; and weight updation.
**Epoch**: the model has performed an epoch, if gradients for whole training data are calculated and the weights are updated. While training a model, it means that multiple epochs are being performed on top of training data.

- **Batch Size**: the **entire training dataset is divided into smaller subsets** or batches and pass it to the network. **The network calculates the Gradient for the whole Training data and updates the weights**. Each batch contains a fixed number of data samples
- **Parameter Update**: The model's parameters (weights and biases) are updated based on the average gradient of the loss function calculated over the entire batch of data.
- **Advantages**:
    - **Speed**: As the whole data is passed as a batch, the network would process the data at once in a vectorized way.
    - Batch gradient descent typically converges to a **more accurate solution** because it uses a comprehensive view of the dataset in each update.
    - It is **well-suited for parallel processing** on hardware like GPUs, making it efficient for large-scale training.
- **Disadvantages**:
    - **Poor learning in initial phases**
    - **High memory consuption**.

**Stochastic Gradient Descent (SGD):**
- **Batch Size**: the **batch size is set to 1**, meaning that **each parameter update is based on a single data sample from the training dataset**. 
- **Parameter Update**: The model's parameters are **updated for each individual data sample**. It computes the gradient of the loss function for one sample and updates the parameters immediately.
- **Advantages**:
    - SGD has **low memory consuption** as it updates the model parameters more frequently. It just loads one data point at a time, so there is no restriction how big the data can be.
    - It can **escape local minima more easily** and **explore a wider range of the loss landscape**, potentially leading to better convergence.
- **Disadvantages**:
    - **Low speed** as there is no batch processing nor vectorization, which means it's very slow to train.
    - **Over learning** as the weights are updated very frequently.
    - It can be **noisy and result in high variance in parameter updates** because it's based on individual samples. This can make convergence **less stable**.
    - It may **require more iterations to converge to a good solution**, and the final solution may not be as precise as that obtained with batch gradient descent.

> Wouldn't it be nice to have the speed of batch gradient descent and memory optimization of SGD?

**Mini-Batch Gradient Descent**:
- Mini-batch gradient descent combines aspects of both batch and stochastic gradient descent. **Instead of using the entire dataset or just one sample, it divides the data into small, fixed-size batches. Parameter updates are based on the average gradient over each mini-batch**.
- Mini-batch gradient descent is a common choice for training neural networks in practice. It balances the computational efficiency of stochastic gradient descent with the stability of batch gradient descent.
- **Advantages**: **Low memory consuption, high speed**. It has lower chances of overfitting.

|       | Stochastic | Mini-Batch | Batch     |
| :---        |    :----:   |    :----:   |          ---: |
| Velocity      | 👍       | 👍       | 👎   |
| Accuracy   | 👎        | 👍       | 👍      |
| Updates   | Every time        | Every mini-batch        | Once       |

___

#<a name="section-4">
#### **Optimizers**
</a>

Optimizers are used for improving speed and performance for training a specific model. **Vanilla gradient descent ensures our model is converging to the optimal weights**, but it does not ensure the speed of convergence. One parameter that impacts the speed of convergence is the learning rate.
- **Learning rate** is multiplied to the gradients before updating weights.

![image](https://miro.medium.com/v2/resize:fit:1400/1*An4tZEyQAYgPAZl396JzWg.png)

**Higher learning rate converges quicker but never gets to the optimal loss**, it just jumps around it. On the other hand, lower learning rate would take a lot of time to learn. It's necessary to choose which learning rate is just right to get the job done. 

___

#<a name="section-5">
#### **Dropout**
</a>

**Dropout refers to deactivating (dropping out) or removing randomly  a portion of neurons or units during training** from a hidden layer in a neural network to **avoid overfitting**. This process involves setting a fraction of neuron activations to zero at each training iteration, effectively 'dropping out' some of the connections. **Regularization reduces overfitting by adding penalty to all features which are useless**. Dropout is an approach to regularization in neural networks which helps in independent learning of the neurons.

The main idea behind dropout is to introduce randomness in the network during training, which discourages neurons from relying too heavily on specific features or connections. This regularization technique promotes better generalization to unseen data and helps prevent the model from fitting the training data noise. 

**Common observation while using dropout for avoiding overfitting in neural networks**:
1. Dropout forces the neural network to learn more robust features that are useful for predictive analytics;
2. Dropout helps us to reduce the training time required for the neural network;
3. choosing the right value for dropout is crucial to get good results.

**Values for Dropout**
- Values for dropout vary from 0 to 1;
- If zero is specified, it means no dropout is required.
- 0.2 means 20% of all neurons from hidden layer is removed.
- 0.5 means 50% of all the neurons are removed at random from the specified hidden layer.

___

#<a name="section-6">
#### **Hyper parameter tuning in neural networks**
</a>

Important parameters to be considered while tuning:
- Number of layers in a neural network;
- The dropout value;
- Learning rate of the neural network;
- Batch size.

**Choosing the right number of hidden layers in a neural network**
- **Medium sized data set**: choose **3 to 5** hidden layers.
- Neural networks might suffer from **underfitting** if the hidden layers are less than 2.
- Similarly, the neural network can suffer from **overfitting** if the number of hidden layers are too high.
- For bigger problems, slowly increase the hidden layers and check see if increasing hidden layers optimizes the neural network.

**How to choose the right value for dropout in neural networks?**
- Most of the times, the value of **0.2 to 0.4** is the most preferable for **dropout**, which means deactivating 20 to 40% of the neurons from the hidden layers.
- Always try to maintain a dropout value **lesser than 0.5**. As higher than 0.5 value for dropout can lead to **overfitting**.
- **Dropout values less than 0.2** value will have no or little significance.

> Start with an initial dropout of 0.2 and check the loss graphs. If the validation loss way too less than training, then it can increase dropout to 0.3 or 0.4 abd recheck the results.

**How to choose the learning rate for neural networks?**
- Optimizing the value of a learning rate is considered to be the **most important step in optimization of neural network**.
- **if the learning rate is low**, then training is more reliable, but optimization will take a **lot of time** because steps towards the minimum of the loss function are tiny; and if the **learning rate is high**, then training may **not converge or even diverge**. Weight changes can be so big that the optimizer overshoots minimum and makes the less worse.
- Most **common learning rates in neural networks are 0.01 or 0.001**.

**How to select the Batch size in neural networks?**
- The selection of batch size depends on RAM or GPU memory. Them more the RAM, the higher the batch size is possible to fit on the memory. Generally, **batch size of 32 is good**, but it should be tried batch sizes of 64, 128, and 256.
- **When to choose high batch size and low batch size?** 
    -If the **problem statement is complex, having low batch size works better**. Low batch size and mini batch gradient descent will have more weight updates which ensures the weights are better tuned.
    - If the **problem statement is simple, try to have the maximum batch size** that can fit in the memory.

___

#<a name="section-7">
#### **Batch normalization**
</a>

Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.

For example, let's suppose I have two features: **age and salary**. Their scales would be 1 to 100 and 10000 to a big number, respectively. For non-normalized values, I will have elongated loss curves where there is a zigzag while performing gradient descent. On the other hand, for normalized values (from 1 to 0) is a straight path towards the center that is optimal loss.

**Advantages of batch normalization**:

![image](https://learnopencv.com/wp-content/uploads/2018/07/ModelAccuracy.png)

- **Faster speed of training**. It would take lesses steps to reach the optimal loss compared to non-normalized;
- **Better accuracy/performance**. When there is less zigzag, it's highly likely that we stabilize at lesses loss in a better way compared to non-normalized;

**Batch normalization also adds two parameters to each layer, mean and standard deviation**. These parameters are calculated while training, and they are used to normalize data while performing real time predictions.