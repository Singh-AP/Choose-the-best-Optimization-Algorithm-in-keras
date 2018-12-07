# Choose-the-best-Optimization-Algorithm-in-keras
Its a prototype of the general idea of quick managing of a Deep Learning project by choosing the best optimization algorithm in order to save time and resources lost due to poor decision of learning rate and choice of algorithm

Hi, let's get staright to the algorithm to choose algorithm:

    For each optimisation algorithm  {
                         For i in range(-5,1){
                                        Run 1 epoch with learning rate 10i 
                                }
                   Choose the value of i which gives the best result
                   Run 50 epochs with the 10i as learning rate and store results.
            }
    Compare the results and choose the best performer. 
   
 
 Here, as proof of concept, we show the results of our algorithm on the [CIFAR 10](https://www.cs.toronto.edu/~kriz/cifar.html).
 
 # Now, 
  There are 7 optimization Algorithms we're concerned with becasue they are present in the    [Keras.Optimizers](https://keras.io/optimizers/) class. Namely
   
   1. [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
   
   2. [RMS Prop](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
   
   3.  [ADAGRAD](https://arxiv.org/abs/1705.08292)
  
   4. [ADADELTA](https://arxiv.org/abs/1212.5701)
  
   5. [ADAM](https://arxiv.org/abs/1412.6980)
   
   6. [ADAMAX](https://arxiv.org/abs/1412.6980)
   
   7. [NADAM](http://cs229.stanford.edu/proj2015/054_report.pdf)
   
# As an example
We use the following CNN model as described [here](https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/ ):

![](https://github.com/TheIndianCoder/Choose-the-best-Optimization-Algorithm-in-keras/blob/master/utils/images/git1.png?raw=true)



# Step 1
  
  Lets import the required libraries
  

# Step 2

1. Import the **CIFAR 10 dataset**:

2. Although the dataset has 60,000 images, we reuire only 20,000 for demonstration purpose

3. Also, we reshape the input dataset to divide it into **X_train ,y_train,X_test and y_test**


# Step 3
 **Create CNN models for each of the 7 optimization algorithms**
  
   Each moel returns the accuracy, time taken and a [Keras History object](https://keras.io/callbacks/#history) which we'll need to plot the graphs.
   
   
After Compiling the models successfully, we now decide which learning rate to choose for each Optimization algorithm we choose the best from

                    [0.00001, 0.0001, 0.001, 0.01, 0.1 , 1, 10] 
by training each model for one epoch and comparing the results

# Step 4:
  Now that we have the ebst learning rates.,
        we choose the best optimization algorithm among the seven
        
# Step 5:
   **Plot the result**
            The history objects will come in handy now
            
            
# Great..! The best optimisation algorithm can now be chosen easily, just by a look at the graph!

Thanks to **Google Colab** for the online GPUs for training.

Again, the CNN model used here as an example has been taken from  [Jason Brownie at Machine Learning Mastery website](https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/ ),.
Although, this algorithm can be applied to any model for faster decision making about various factors. 

