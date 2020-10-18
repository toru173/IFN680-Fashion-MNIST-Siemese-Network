# IFN680-Fashion MNIST Siemese Network
 A Siemese Neural Network to classify objects in the Fashion MNIST dataset as part of the fulfillments for Queensland University of Technology's IFN680 (Advanced Topics in Artificial Intelligence).

The network attempts to compare two objects from the Fashion MNIST dataset (https://github.com/zalandoresearch/fashion-mnist), and return whether or not they are from the same catagory. For example, when shown two different shoes, the network will return that they are the same sort of fashion item. When shown a shoe and a handbag, however, the network will return that they are different items.

Due to a logical error in the programming the network did not gain accuracy with  further training. The results can be seen below:

![](https://github.com/toru173/IFN680-Fashion-MNIST-Siemese-Network/blob/main/Figure_1.png)

When this script is run stand-alone rather than being included in another, larger script, the above experiments are run. Experiment One increases the number of epochs from 1 to num_epochs, and graphs the accuracy of the model accordingly. Experiment Two examins the effect of increasing the size each batch of images on which the model is trained simultanously, up to a maximum size of of max_batch_size. Experiment Three focuses on the total number of unique pair samples used to train the model, up to max_sample_size. For each experiment the result is graphed and also output in CSV format to the directory in which the script is run.