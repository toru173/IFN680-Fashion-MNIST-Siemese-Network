"""
    my_solution.py is the solution for Assignment Two in IFN680.
    
    Name: Thomas Mahoney
    Student Number: n9919783
    
    The script is designed as an importable module that allows the user
    to create an instance of class SiameseModel that will return a Siamese
    Nueral Network based on the Keras framework. For information on how to
    evaluate the performance of the network, please scroll down to the line
    begining with 'if __name__ == "__main__" :' (line 239)
    
"""

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow import boolean_mask
from matplotlib import pyplot
import keras.backend as tensorMath # https://keras.io/backend/
import numpy as np
from datetime import datetime

class SiameseModel(Model) :
    
    def __init__(self, shape, optimiser = "rmsprop", metrics = ["accuracy"]) :
        """
            Instantiate the class. Extend the Model class from keras.models
            
            Arguments:
                - shape: the shape of the input tensor
                - optimiser: Keras optimiser to use when training model. Default is "rmsprop"
                - metrics: Keras metrics on which to train. Default is "accuracy"
            Returns:
                - None
        """
        # Instantiate model
        super(SiameseModel, self).__init__()
        self.shape = shape
        self.model = self.createSiameseNetwork(optimiser, metrics)

    def euclideanDistanceShape(self, shapes) :
        """
            Return the shape of a Euclidean Distance tensor. Used in
            contrastiveLoss().
            
            Arguments:
                - shapes: One or more tensors
            Returns:
                - A tuple containing the shape of the first tensor
        """
        # Returns the same of the first tensor in shapes. Assume the shape of
        # all tensors in shapes are the same.
        return (shapes[0][0], 1)

    def euclideanDistance(self, tensors) :
        """
            Calculates the Euclidean distance between two tensors.
            
            Arguemnts:
                - tensors: An object containing one or more tensors
            Return
                - A tensor representing the Euclidean Distance of all the input tensors
        """
        # Returns the euclidean distance of two tensors:
        # D = sum((tensor_x - tensor_y) ** 2) ** 0.5
        # https://en.wikipedia.org/wiki/Euclidean_distance
        # Need to use Keras math libraries when performing operations on tensors!
        # Solution outlined in the Keras documentation at
        # https://keras.io/examples/mnist_siamese/
        tensor_x, tensor_y = tensors
        return tensorMath.sqrt(tensorMath.sum(tensorMath.square(tensor_x - tensor_y),
                                              axis=1, keepdims=True))
        
    def contrastiveLoss(self, truth, prediction, margin = 1) :
        """
            Computes the Contrastive Loss, as outlined in the assignment documentation
            on page four.
            
            Arguments:
                - truth: A tensor containing truth-value pairs, usually of the format [1, 0]
                - prediction: A tensor containing the Euclidean distance between the two pairs
            Returns:
                - The tensor representation of the function outlined in the assignment
                  documentation on page 4
        """
        # euclideanDistance passed to the loss function in the position of "prediction"
        # and whether the pairs represent a similar (1) or dissimilar (0) pair passed
        # as "truth" as this is how loss functions interface with the Keras framework
        # Loss = 0.5 * euclideanDistance(tensor_x, tensor_y) ** 2 if tensor_x := tensor_y
        # Loss = 0.5 * maximum(margin - euclideanDistance(tensor_x, tensor_y), 0) ** 2
        # otherwise.
        # Unable to scale a tensor with the available libraries. Instead, return
        # the mean of two tensors, one multiplied by the equivilence variable and
        # one multiplied by the inverse of this variable. This will result in
        # either (0 + negative_pair) = 0.5 * negative_pair or (positive_pair + 0)
        # = 0.5 * positive_pair. Solution outlined in the Keras documentation at
        # https://keras.io/examples/mnist_siamese/
        positive_pair = tensorMath.square(prediction)
        negative_pair = tensorMath.square(tensorMath.maximum(margin - prediction, 0))
        return tensorMath.mean(truth * positive_pair + (1 - truth) * negative_pair)
        
    def createMonozygoticNetwork(self) :
        """
            Creates a single neural network with weights and layers outlined
            as per below.
            
            Arguments:
                - None
            Returns:
                - A Model object containing the neural network
        """
        # Create a single network with a particular structure and weights
        # From https://keras.io/getting-started/functional-api-guide/
        # Dimensionality of the output space and activation function found
        # experimentally based on best performance for unseen pairs
        input = Input(shape = self.shape)
        x = Flatten()(input)
        x = Dense(256, activation='sigmoid')(x)
        x = Dense(256, activation='sigmoid')(x)
        x = Dense(256, activation='sigmoid')(x)
        return Model(input, x)
    
    def createSiameseNetwork(self, optimiser, metrics) :
        """
            Returns a Siamese network model using two representations
            of the model outlined in createMonozygoticNetwork().
            
            Arguments:
                - optimiser: A Keras optimiser to use when training model
                - metrics: Keras metric on which to train
            Returns:
                - A Model object representing the Siamese network model
        """
        # Chang and Eng Bunker were the original Siamese twins
        # https://en.wikipedia.org/wiki/Chang_and_Eng_Bunker
        # Network features named after them in their honour
        base_network = self.createMonozygoticNetwork()
        # Create inputs for each half of the siamese network
        chang_input = Input(shape = self.shape)
        eng_input = Input(shape = self.shape)
        # Extract features using the base network
        chang_features = base_network(chang_input)
        eng_features = base_network(eng_input)
        # Add a final layer that computers the euclidean distance between the
        # two featuresets
        euclideanDistanceLayer = Lambda(self.euclideanDistance,
                                        output_shape = self.euclideanDistanceShape) \
                                        ([chang_features, eng_features])
        # Create the model with input and output layers
        model = Model(name = "Siamese_Network",
                      inputs = [chang_input, eng_input],
                      outputs = euclideanDistanceLayer)
        # Compile the model, then return it to the instantiating method
        model.compile(loss = self.contrastiveLoss,
                      optimizer = optimiser,
                      metrics = metrics)
    
        return model
         

def getBatchOfPairs(training_data, training_labels, batch_size = None) :
    """
        Used to get a set of training and testing pairs with batch size
        batch_size
        
        Arguemnts:
            - training_data: A NumPy array containing a selection of images
                             such as those found in Fashion-MNIST on which
                             to train
            - training_labels: A NumPy array of labels corresponding to each
                               image in the array of training_data
            - batch_size: Number of samples to include in this training batch.
                          Default is the to return a batch of pairs equal in
                          size to the number of objects in training_data
        Returns:
            - A NumPy array of positive and netagive (similar and dissimilar)
              training pairs
            - A NumPy array in the form of [[1, 0], ...] outlining whether
              the pair at the same index are positive (similar) or
              negative (dissimilar)
    """
    # Assume training_data of shape similar to (num_entries, x_dim, y_dim)
    # Don't always assume we want to train against the whole dataset; IE, in
    # the case of very large images we may only be able to hold a small
    # number in memory!
    if batch_size == None : batch_size = training_data.shape[0]
    training_data_shape_x = training_data.shape[1]
    training_data_shape_y = training_data.shape[2]
    # Batch size can only be an even number (positive AND negative pairs)
    if (batch_size % 2) != 0 : raise ValueError("Batch Size must be a multiple of two")
    batch = np.empty((batch_size, 2, training_data_shape_x, \
                      training_data_shape_y), dtype = training_data.dtype)
                      
    positive_or_negative_pair = [1, 0] * (batch_size // 2)
    entries = 0
    for pairs in getPairs(training_data, training_labels, batch_size // 2) :
        batch[entries, 0] = pairs[0]
        batch[entries, 1] = pairs[1]
        entries += 1
        batch[entries, 0] = pairs[0]
        batch[entries, 1] = pairs[2]
        entries += 1
        
    return np.array(batch), np.array(positive_or_negative_pair)
        

def getPairs(training_data, training_labels, num_pairs = -1) :
    """
        Generator used for generating training data. Will find a random
        object from training_data, then find and return a corresponding
        positive (similar) and negative (dissimilar) image using the array
        training_labels. Will generate a maximum of num_pairs pairs before
        raising StopIteration.
    
        Arguments:
            - training_data: An array of objects
            - training_labels: An array dictating what equivilence class
                               each object in training_data belongs to
            - num_pairs: The number of pairs to generate before raising
                         StopIteration. The default is one pair
            Returns:
                - An object randomly selected to be the comparison object
                - Another random object that is in the same equivilence class
                - Another random object that is in a different equivilence class
    """
    # Default is to generate an unlimited number of training pairs
    # Assume training_data of shape similar to (num_entries, x_dim, y_dim)
    max_index = training_labels.shape[0] - 1
    pairs = 0
    while pairs < num_pairs or pairs == 0:
        # Select a random element
        index = np.random.randint(0, max_index)
        candidate_element = training_data[index]
        candidate_label = training_labels[index]
        positive_element = None
        negative_element = None
        while positive_element is None or negative_element is None :
            # We need to find two other elements: one with the same label,
            # a second with a different label. Iterate until we have found
            # both positive and negative examples
            # First find another, random element
            offset = np.random.randint(0, max_index)
            index = (index + offset) % max_index
            if training_labels[index] == candidate_label :
                # If the label is the same as the candidate, add it as a
                # positive example
                positive_element = training_data[index]
            else :
                # Otherwise, add it as a negative example
                negative_element = training_data[index]
        # Return the chosen, random element, and a positive and negative match
        yield candidate_element, positive_element, negative_element
        pairs += 1


def prepareDataset(dataset, labels = None, test_percentage = 0) :
    """
        Given a dataset, divide the dataset such that we only consider the
        objects corresponding to the labels in labels. Return a certain
        portion of the dataset that can be used for training, and another
        for testing
        
        Arguments:
            - dataset: A set of data on which to perform test. MNIST-alike
            - labels: The labels corresponding to the objects in the dataset
                      on which we want to test. Default is None
            - test_percentage: The percentage of the dataset we want to withhold
                               for testing only, and not include in the training
                               dataset. Default is 0%
        Returns:
            - A NumPy array containing images used for training
            - A NumPy array containing images used for testing
            - A NumPy array containing labels corresponding to training images
            - A NumPy array containing labels corresponding to testing images
    """
    test_percentage /= 100.0
    # We can use a generator here to save on memory usage, but we only
    # need to deal with 10 items so use a list instead
    label_indicies = [descriptionToLabel(label) for label in labels]

    (x_train, x_test), (y_train, y_test) = dataset.load_data()
        
    # Scale data to 0 - 1
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_train /= 255
    y_train /= 255
    
    # We need to be able to control what clothing items in Fashion-MNIST we
    # train or test against. This is done by using the boolean_mask() method
    # in tensorflow, then casting to a numpy array
    training_dataset_mask = []
    testing_dataset_mask = []
        
    for label in x_test :
        if label in label_indicies :
            training_dataset_mask.append(True)
        else :
            training_dataset_mask.append(False)
        
    for label in y_test :
        if label in label_indicies :
            testing_dataset_mask.append(True)
        else :
            testing_dataset_mask.append(False)
    
    selected_elements = boolean_mask(x_train, training_dataset_mask).numpy(),\
                        boolean_mask(x_test, training_dataset_mask).numpy(),\
                        boolean_mask(y_train, testing_dataset_mask).numpy(),\
                        boolean_mask(y_test, testing_dataset_mask).numpy()

    # Fashion-MNIST has a pre-selected division for training (60000 images)
    # and testing (10000 images). We want to be able to control this ratio,
    # so the training and testing sets are concatenated then two new sets
    # are created with the appropriate percentage of elements in each
    selected_images = np.concatenate((selected_elements[0],
                                      selected_elements[2]))
    
    selected_labels = np.concatenate((selected_elements[1],
                                      selected_elements[3]))
    
    if test_percentage == 0 :
        return selected_images, None, selected_labels, None
    else :
        return train_test_split(selected_images, selected_labels,
                                test_size = test_percentage)
  

def calculateAccuracy(predicted_values, truth_values, threshold = 0.5) :
    """
        Used to calculate the accuracy of a model
        
        Arguments:
            - predicted_values: An object containing a list or array of
                                Euclidean distances, representing whether or
                                not a pair of images are similar or dissimilar
            - truth_values: An object containing a list or array that dictates
                            whether the trained pair of images were from the
                            same equivilence class; that is, whether they were
                            similar or dissimilar
            - threshold: An abitrary value. Below "threshold", Euclidean Distances
                         are considered 'similar enough;' above this value the
                         difference is considered too high to assume the tensors
                         are from the same equivilance class
        Returns:
            - A float containing the percentage of predictions were accurate
    """
    # Returns the percent of the number of instances where the predicted
    # value is less than the threshold number AND that this is the correct
    # choice; that is, that the euclidean distance of the pair is low enough
    # such that they can be considered the same, and that they are the same
    return 100 * np.mean([predicted_values.ravel() < threshold] == truth_values)
    
    
def labelToDescription(label) :
    """
        Given a label ([0 ... 9]), return the corresponding
        Fashion-MNIST description
        
        Arguments:
            - label: An integer in the range 0 - 9
        Returns:
            - The correspondinding string from the list 'descriptions'
    """
    descriptions = ["top", "trouser", "pullover",
                    "dress", "coat", "sandal",
                    "shirt", "sneaker", "bag",
                    "ankle boot"]
    return descriptions[label]
    

def descriptionToLabel(description) :
    """
        Given a description, return the corresponding label ([0 ... 9])
        
        Arguments:
            - description: A string containing a description of a
                           Fashion-MNIST image
        Returns:
            - An integer that indicates the corresponding label of the description
    """
    descriptions = ["top", "trouser", "pullover",
                    "dress", "coat", "sandal",
                    "shirt", "sneaker", "bag",
                    "ankle boot"]
    return descriptions.index(description)


if __name__ == "__main__" :
    """
        Runs when script is run stand-alone, rather than being imported into a larger
        python script or project.
        
        Arguemnts:
            - None
        Returns:
            -None
    """

    """
        Experimental bounds set out below. When this script is run as a
        stand-alone rather than being included in another, larger script,
        the below experiments are run. Experiment One increases the number of
        epochs from 1 to num_epochs, and graphs the accuracy of the model
        accordingly. Experiment Two examins the effect of increasing the size
        each batch of images on which the model is trained simultanously, up to
        a maximum size of of max_batch_size. Experiment Three focuses on the
        total number of unique pair samples used to train the model, up to
        max_sample_size. For each experiment the result is graphed and also
        output in CSV format to the directory in which the script is run.
        
        PLEASE NOTE: the defaults result in quite long run times. If testing,
        please run only one test (setting the other two variables to 0), or
        set each to very small values. Note that experiment three can require
        a significant amount of RAM for large sample sizes.
        
        The below is written for human-like reading, rather than optimising
        for short execution, low line count or tight loops and is thus extremely
        verbose. A lot of unnecessary progress and information is written to
        the command line during the execution of the script!
    """
    
    print("[INFO] Setting up experimental bounds..." )
    
    """
        Set experimental bounts here. To stop an experiment from running,
        set it's associated variable to 0
    """
    
    
    num_epochs = 128            # Experiment One. For short run, recommend <= 5
    max_batch_size = 2 ** 10    # Experiment Two. For short run, recommned <= 2 ** 6
    max_sample_size = 2 ** 20   # Experiment Three. For short run, recommend <= 2 ** 14
    
    
    # Easy way to create a list of the powers of two less
    # than or equal to the maximum stipulated. Max available is 2 ** 32
    # for performance reasons
    batches = [2 ** i for i in range(32) if 2 ** i <= max_batch_size]
    # Sample Size must be in the range of 2 ** 6 = 64 to 2 ** 32 = 4,294,967,296
    # We use 20% of our samples for testing, so we must have at least 64 samples
    # to give a meaningful test set size
    samples = [2 ** i for i in range(6, 32) if 2 ** i <= max_sample_size]
    
    training_dataset_labels = ["top", "trouser", "pullover",
                               "coat", "sandal", "ankle boot"]
    validation_dataset_labels = ["dress", "shirt", "sneaker", "bag"]
    
    print("[INFO] Creating training and testing datasets...")
    
    training_images, testing_images, training_labels, testing_lables = \
        prepareDataset(fashion_mnist, training_dataset_labels,
                       test_percentage = 20)

    print("[INFO] Creating validation datasets...")

    validation_images, x, validation_labels, y = \
        prepareDataset(fashion_mnist,
                       training_dataset_labels + validation_dataset_labels)
    
    print("[INFO] Creating unseen datasets...")
    
    unseen_images, x, unseen_labels, y = \
        prepareDataset(fashion_mnist, validation_dataset_labels)
    
    print("[INFO] Creating training pairs...")
    
    training_pairs, training_equivalences = \
        getBatchOfPairs(training_images, training_labels)
    
    print("[INFO] Creating testing pairs...")
    
    testing_pairs, testing_equivalences = \
        getBatchOfPairs(testing_images, testing_lables)
    
    print("[INFO] Creating validation pairs...")
    
    validation_pairs, validation_equivalences = \
        getBatchOfPairs(validation_images, validation_labels)
    
    print("[INFO] Creating unseen pairs...")
    
    unseen_pairs, unseen_equivalences = \
        getBatchOfPairs(unseen_images, unseen_labels)
    
    # Create arrays to store experimental results
    epoch_results = np.empty([num_epochs, 5])
    batch_size_results = np.empty([len(batches), 5])
    sample_size_results = np.empty([len(samples), 5])
    
    epoch_results[ : , 0] = [i + 1 for i in range(num_epochs)]
    batch_size_results[ : , 0] = batches
    sample_size_results[ : , 0] = samples
    
    # EXPERIMENT ONE: Training Accuracy vs Number of Epochs
    if num_epochs : print("[INFO] ============ EXPERIMENT ONE ============")
    for epoch in range(1, num_epochs + 1) :
    
        if epoch == 1 : print("[INFO] Training network for " + str(epoch) + " epoch")
        else : print("[INFO] Training network for " + str(epoch) + " epochs")

        siamese_network = SiameseModel(shape = training_images[0].shape)

        siamese_network.model.fit([training_pairs[ : , 0],
                                   training_pairs[ : , 1]],
                                   training_equivalences,
                                   batch_size = 784,
                                   epochs = epoch,
                                   validation_data = ([
                                   testing_pairs[ : , 0],
                                   testing_pairs[ : , 1]],
                                   testing_equivalences))
    
    
        print("[INFO] Calculating accuracy for training pairs...")
        predictions = siamese_network.model.predict([training_pairs[ : , 0],
                                                     training_pairs[ : , 1]])
        epoch_results[epoch - 1, 1] = calculateAccuracy(predictions,
                                                        training_equivalences)
        
        print("[INFO] Calculating accuracy for testing pairs...")
        predictions = siamese_network.model.predict([testing_pairs[:, 0],
                                                     testing_pairs[:, 1]])
        epoch_results[epoch - 1, 2] = calculateAccuracy(predictions,
                                                        testing_equivalences)
        
        print("[INFO] Calculating accuracy for validation pairs...")
        predictions = siamese_network.model.predict([validation_pairs[:, 0],
                                                     validation_pairs[:, 1]])
        epoch_results[epoch - 1, 3] = calculateAccuracy(predictions,
                                                        validation_equivalences)
        
        print("[INFO] Calculating accuracy for unseen pairs...")
        predictions = siamese_network.model.predict([unseen_pairs[:, 0],
                                                     unseen_pairs[:, 1]])
        epoch_results[epoch - 1, 4] = calculateAccuracy(predictions,
                                                        unseen_equivalences)
    
    # EXPERIMENT TWO: Training Accuracy vs Batch Size
    if batches : print("[INFO] ============ EXPERIMENT TWO ============")
    for batch in range(len(batches)) :
        # As we need the size and the index, this is the easiest way
        # to iterate through the list
        
        print("[INFO] Training with batch size of " + str(batches[batch]))
        
        siamese_network = SiameseModel(shape = training_images[0].shape)
        siamese_network.model.fit([training_pairs[ : , 0],
                                   training_pairs[ : , 1]],
                                   training_equivalences,
                                   batch_size = batches[batch],
                                   epochs = 128,
                                   validation_data = ([
                                   testing_pairs[ : , 0],
                                   testing_pairs[ : , 1]],
                                   testing_equivalences))
                                  
        print("[INFO] Calculating accuracy for training pairs...")
        predictions = siamese_network.model.predict([training_pairs[ : , 0],
                                                     training_pairs[ : , 1]])
        batch_size_results[batch, 1] = calculateAccuracy(predictions,
                                                         training_equivalences)

        print("[INFO] Calculating accuracy for testing pairs...")
        predictions = siamese_network.model.predict([testing_pairs[:, 0],
                                                     testing_pairs[:, 1]])
        batch_size_results[batch, 2] = calculateAccuracy(predictions,
                                                         testing_equivalences)

        print("[INFO] Calculating accuracy for validation pairs...")
        predictions = siamese_network.model.predict([validation_pairs[:, 0],
                                                     validation_pairs[:, 1]])
        batch_size_results[batch, 3] = calculateAccuracy(predictions,
                                                         validation_equivalences)

        print("[INFO] Calculating accuracy for unseen pairs...")
        predictions = siamese_network.model.predict([unseen_pairs[:, 0],
                                                     unseen_pairs[:, 1]])
        batch_size_results[batch, 4] = calculateAccuracy(predictions,
                                                         unseen_equivalences)

    # EXPERIMENT THREE: Training Accuracy vs Number of Samples
    if samples : print("[INFO] ============ EXPERIMENT THREE ===========")
    for sample in range(len(samples)) :
        # As we need the size and the index, this is the easiest way
        # to iterate through the list
        # Training sample size must be even; it always will be, as samples[]
        # are all powers of two
        training_sample_size = samples[sample]
        # Testing sample saize is 20% of training sample size
        testing_sample_size = samples[sample] // 5
        # testing_sample_size must be even
        if testing_sample_size %2 != 0 : testing_sample_size -= 1
        print("[INFO] Creating " + str(training_sample_size) + " training pairs ")
        
        training_pairs, training_equivalences = \
            getBatchOfPairs(training_images, training_labels, training_sample_size)
        
        print("[INFO] Creating " + str(testing_sample_size) + " testing pairs...")
        
        testing_pairs, testing_equivalences = \
            getBatchOfPairs(testing_images, testing_lables, testing_sample_size)
            
        if sample == 0 : print("[INFO] Training with " + str(training_sample_size) + \
                                " sample")
        else : print("[INFO] Training with " + str(training_sample_size) + " samples")
        
        siamese_network = SiameseModel(shape = training_images[0].shape)
        siamese_network.model.fit([training_pairs[ : , 0],
                                   training_pairs[ : , 1]],
                                   training_equivalences,
                                   batch_size = 784,
                                   epochs = 128,
                                   validation_data = ([
                                   testing_pairs[ : , 0],
                                   testing_pairs[ : , 1]],
                                   testing_equivalences))
                                  
        print("[INFO] Calculating accuracy for training pairs...")
        predictions = siamese_network.model.predict([training_pairs[ : , 0],
                                                     training_pairs[ : , 1]])
        sample_size_results[sample, 1] = calculateAccuracy(predictions,
                                                         training_equivalences)

        print("[INFO] Calculating accuracy for testing pairs...")
        predictions = siamese_network.model.predict([testing_pairs[:, 0],
                                                     testing_pairs[:, 1]])
        sample_size_results[sample, 2] = calculateAccuracy(predictions,
                                                         testing_equivalences)

        print("[INFO] Calculating accuracy for validation pairs...")
        predictions = siamese_network.model.predict([validation_pairs[:, 0],
                                                     validation_pairs[:, 1]])
        sample_size_results[sample, 3] = calculateAccuracy(predictions,
                                                         validation_equivalences)

        print("[INFO] Calculating accuracy for unseen pairs...")
        predictions = siamese_network.model.predict([unseen_pairs[:, 0],
                                                     unseen_pairs[:, 1]])
        sample_size_results[sample, 4] = calculateAccuracy(predictions,
                                                         unseen_equivalences)
    
    # Write out results to a CSV in the same directory as the script,
    # with each experimental result named accordingly. Result CSV files
    # named with completion time appended to experiment name.
    print("[INFO] Writing results to file...")
    completion_time = datetime.now().strftime("%Y-%m-%d %H-%M")
    
    if num_epochs : 
        np.savetxt("Experiment One - Varying Number of Epochs " + \
                   completion_time + ".csv",
                   epoch_results, delimiter = ',', fmt = "%f",
                   header = "Number of Epochs," + \
                            "Training Pairs Accuracy," + \
                            "Testing Pairs Accuracy," + \
                            "Validation Pairs Accuracy," + \
                            "Unseen Pairs Accuracy",
                  comments="")
    
    if batches :
        np.savetxt("Experiment Two - Varying Batch Size " +\
                   completion_time + ".csv",
                   batch_size_results, delimiter = ',', fmt = "%f",
                   header = "Batch Size," + \
                            "Training Pairs Accuracy," + \
                            "Testing Pairs Accuracy," + \
                            "Validation Pairs Accuracy," + \
                            "Unseen Pairs Accuracy",
                   comments="")
    
    if samples :
        np.savetxt("Experiment Three - Varying Training Sample Size " + \
                   completion_time + ".csv",
                   sample_size_results, delimiter = ',', fmt = "%f",
                   header = "Sample Size," + \
                            "Training Pairs Accuracy," + \
                            "Testing Pairs Accuracy," + \
                            "Validation Pairs Accuracy," + \
                            "Unseen Pairs Accuracy",
                   comments="")

    # Display results as a matplotlib.pyplot object. All results are displayed
    # in the same object, ordered by experiment number
    
    num_experiments = 0
    
    if num_epochs : num_experiments += 1
    if batches : num_experiments += 1
    if samples : num_experiments += 1
    
    position = 0
    if num_epochs :
        position += 1
        x_axis = [i + 1 for i in range(num_epochs)]
        
        pyplot.subplot(num_experiments, 1, position)
        pyplot.plot(x_axis, epoch_results[ : , 1],
                    label = "Training Pairs Accuracy")
        pyplot.plot(x_axis, epoch_results[ : , 2],
                    label = "Testing Pairs Accuracy")
        pyplot.plot(x_axis, epoch_results[ : , 3],
                    label = "Validation Pairs Accuracy")
        pyplot.plot(x_axis, epoch_results[ : , 4],
                    label = "Unseen Pairs Accuracy")
        pyplot.xlabel("Number of Training Epochs")
        pyplot.ylabel("Accuracy (Percent)")
        pyplot.legend(loc="upper right")
        
    if max_batch_size :
        position += 1
        x_axis = batches
        pyplot.subplot(num_experiments, 1, position)
        pyplot.plot(x_axis, batch_size_results[ : , 1],
                    label = "Training Pairs Accuracy")
        pyplot.plot(x_axis, batch_size_results[ : , 2],
                    label = "Testing Pairs Accuracy")
        pyplot.plot(x_axis, batch_size_results[ : , 3],
                    label = "Validation Pairs Accuracy")
        pyplot.plot(x_axis, batch_size_results[ : , 4],
                    label = "Unseen Pairs Accuracy")
        pyplot.xlabel("Batch Size")
        pyplot.ylabel("Accuracy (Percent)")
        pyplot.legend(loc="upper right")

    if max_sample_size :
        position += 1
        x_axis = samples
        pyplot.subplot(num_experiments, 1, position)
        pyplot.plot(x_axis, sample_size_results[ : , 1],
                    label = "Training Pairs Accuracy")
        pyplot.plot(x_axis, sample_size_results[ : , 2],
                    label = "Testing Pairs Accuracy")
        pyplot.plot(x_axis, sample_size_results[ : , 3],
                    label = "Validation Pairs Accuracy")
        pyplot.plot(x_axis, sample_size_results[ : , 4],
                    label = "Unseen Pairs Accuracy")
        pyplot.xlabel("Number of Samples")
        pyplot.ylabel("Accuracy (Percent)")
        pyplot.legend(loc="upper right")

    if num_experiments :
        pyplot.tight_layout()
        pyplot.show()
