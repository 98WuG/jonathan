from __future__ import absolute_import

import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from preprocess import cut_random_cubes, normalize, zero_center, display_ct_pet_processed
import time
from datetime import datetime


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_size = 2
        self.checkpoint = 1

        # First convolution layer, relu, and normalization
        self.conv1 = tf.keras.layers.Conv3D(21, 3, padding='same')
        self.conv1_norm = tfa.layers.normalizations.InstanceNormalization()

        # First residual block
        self.res11 = tf.keras.layers.Conv3D(21, 3, padding='same')
        self.res11_norm = tfa.layers.normalizations.InstanceNormalization()
        self.res12 = tf.keras.layers.Conv3D(21, 3, padding='same')
        self.res12_norm = tfa.layers.normalizations.InstanceNormalization()

        # Second convolution layer, stride 2
        self.conv2 = tf.keras.layers.Conv3D(42, 3, strides=(2, 2, 2), padding='same')
        self.conv2_norm = tfa.layers.normalizations.InstanceNormalization()

        # Second residual block
        self.res21 = tf.keras.layers.Conv3D(42, 3, padding='same')
        self.res21_norm = tfa.layers.normalizations.InstanceNormalization()
        self.res22 = tf.keras.layers.Conv3D(42, 3, padding='same')
        self.res22_norm = tfa.layers.normalizations.InstanceNormalization()

        # Third convolution layer, stride 2
        self.conv3 = tf.keras.layers.Conv3D(84, 3, strides=(2, 2, 2), padding='same')
        self.conv3_norm = tfa.layers.normalizations.InstanceNormalization()

        # Third residual block
        self.res31 = tf.keras.layers.Conv3D(84, 3, padding='same')
        self.res31_norm = tfa.layers.normalizations.InstanceNormalization()
        self.res32 = tf.keras.layers.Conv3D(84, 3, padding='same')
        self.res32_norm = tfa.layers.normalizations.InstanceNormalization()

        # Fourth convolution layer, stride 2
        self.conv4 = tf.keras.layers.Conv3D(168, 3, strides=(2, 2, 2), padding='same')
        self.conv4_norm = tfa.layers.normalizations.InstanceNormalization()

        # Fourth residual block
        self.res41 = tf.keras.layers.Conv3D(168, 3, padding='same')
        self.res41_norm = tfa.layers.normalizations.InstanceNormalization()
        self.res42 = tf.keras.layers.Conv3D(168, 3, padding='same')
        self.res42_norm = tfa.layers.normalizations.InstanceNormalization()

        # Fifth convolution layer, stride 2
        self.conv5 = tf.keras.layers.Conv3D(336, 3, strides=(2, 2, 2), padding='same')
        self.conv5_norm = tfa.layers.normalizations.InstanceNormalization()

        # Fifth residual block
        self.res51 = tf.keras.layers.Conv3D(336, 3, padding='same')
        self.res51_norm = tfa.layers.normalizations.InstanceNormalization()
        self.res52 = tf.keras.layers.Conv3D(336, 3, padding='same')
        self.res52_norm = tfa.layers.normalizations.InstanceNormalization()

        # First upsample block
        self.up1_upsample = tf.keras.layers.UpSampling3D(2)
        self.up1 = tf.keras.layers.Conv3D(168, 3, padding='same')
        self.up1_norm = tfa.layers.normalizations.InstanceNormalization()

        # First localization module
        self.loc11 = tf.keras.layers.Conv3D(336, 3, padding='same')
        self.loc11_norm = tfa.layers.normalizations.InstanceNormalization()
        self.loc12 = tf.keras.layers.Conv3D(168, 1, padding='same')
        self.loc12_norm = tfa.layers.normalizations.InstanceNormalization()

        # Second upsample block
        self.up2_upsample = tf.keras.layers.UpSampling3D(2)
        self.up2 = tf.keras.layers.Conv3D(84, 3, padding='same')
        self.up2_norm = tfa.layers.normalizations.InstanceNormalization()

        # Second localization module
        self.loc21 = tf.keras.layers.Conv3D(168, 3, padding='same')
        self.loc21_norm = tfa.layers.normalizations.InstanceNormalization()
        self.loc22 = tf.keras.layers.Conv3D(84, 1, padding='same')
        self.loc22_norm = tfa.layers.normalizations.InstanceNormalization()

        # Third upsample block
        self.up3_upsample = tf.keras.layers.UpSampling3D(2)
        self.up3 = tf.keras.layers.Conv3D(42, 3, padding='same')
        self.up3_norm = tfa.layers.normalizations.InstanceNormalization()

        # Third localization module
        self.loc31 = tf.keras.layers.Conv3D(84, 3, padding='same')
        self.loc31_norm = tfa.layers.normalizations.InstanceNormalization()
        self.loc32 = tf.keras.layers.Conv3D(42, 1, padding='same')
        self.loc32_norm = tfa.layers.normalizations.InstanceNormalization()

        # Fourth upsample block
        self.up4_upsample = tf.keras.layers.UpSampling3D(2)
        self.up4 = tf.keras.layers.Conv3D(21, 3, padding='same')
        self.up4_norm = tfa.layers.normalizations.InstanceNormalization()

        # Last convolution layer
        self.conv6 = tf.keras.layers.Conv3D(42, 3, padding='same')
        self.conv6_norm = tfa.layers.normalizations.InstanceNormalization()

        # Segmentation layer 1
        self.seg1 = tf.keras.layers.Conv3D(1, 1, padding='same')
        self.seg1_softmax = tf.keras.layers.Softmax

        # Segmentation layer 2
        self.seg2 = tf.keras.layers.Conv3D(1, 1, padding='same')
        self.seg2_softmax = tf.keras.layers.Softmax

        # Segmentation layer 3
        self.seg3 = tf.keras.layers.Conv3D(1, 1, padding='same')
        self.seg3_softmax = tf.keras.layers.Softmax

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """

        # Top layer
        conv1_out = tf.nn.leaky_relu(self.conv1_norm(self.conv1(inputs)))

        res11_out = tf.nn.leaky_relu(self.res11_norm(self.res11(conv1_out)))
        res12_out = tf.nn.leaky_relu(self.res12_norm(self.res12(res11_out)))

        layer1_out = conv1_out + res12_out

        # Second layer
        conv2_out = tf.nn.leaky_relu(self.conv2_norm(self.conv2(layer1_out)))

        res21_out = tf.nn.leaky_relu(self.res21_norm(self.res21(conv2_out)))
        res22_out = tf.nn.leaky_relu(self.res22_norm(self.res22(res21_out)))

        layer2_out = conv2_out + res22_out

        # Third layer
        conv3_out = tf.nn.leaky_relu(self.conv3_norm(self.conv3(layer2_out)))

        res31_out = tf.nn.leaky_relu(self.res31_norm(self.res31(conv3_out)))
        res32_out = tf.nn.leaky_relu(self.res32_norm(self.res32(res31_out)))

        layer3_out = conv3_out + res32_out

        # Fourth layer
        conv4_out = tf.nn.leaky_relu(self.conv4_norm(self.conv4(layer3_out)))

        res41_out = tf.nn.leaky_relu(self.res41_norm(self.res41(conv4_out)))
        res42_out = tf.nn.leaky_relu(self.res42_norm(self.res42(res41_out)))

        layer4_out = conv4_out + res42_out

        # Fifth layer
        conv5_out = tf.nn.leaky_relu(self.conv5_norm(self.conv5(layer4_out)))

        res51_out = tf.nn.leaky_relu(self.res51_norm(self.res51(conv5_out)))
        res52_out = tf.nn.leaky_relu(self.res52_norm(self.res52(res51_out)))

        layer5_out = conv5_out + res52_out

        # First upsample module
        up1_out = tf.nn.leaky_relu(self.up1_norm(self.up1(self.up1_upsample(layer5_out))))

        # First localization module
        up1_concat = tf.keras.layers.concatenate([up1_out, layer4_out], 4)
        loc11_out = tf.nn.leaky_relu(self.loc11_norm(self.loc11(up1_concat)))
        loc12_out = tf.nn.leaky_relu(self.loc12_norm(self.loc12(loc11_out)))

        # Second upsample module
        up2_out = tf.nn.leaky_relu(self.up2_norm(self.up2(self.up2_upsample(loc12_out))))

        # Second localization module
        up2_concat = tf.keras.layers.concatenate([up2_out, layer3_out], 4)
        loc21_out = tf.nn.leaky_relu(self.loc21_norm(self.loc21(up2_concat)))
        loc22_out = tf.nn.leaky_relu(self.loc22_norm(self.loc22(loc21_out)))

        # Third upsample module
        up3_out = tf.nn.leaky_relu(self.up3_norm(self.up3(self.up3_upsample(loc22_out))))

        # Third localization module
        up3_concat = tf.keras.layers.concatenate([up3_out, layer2_out], 4)
        loc31_out = tf.nn.leaky_relu(self.loc31_norm(self.loc31(up3_concat)))
        loc32_out = tf.nn.leaky_relu(self.loc32_norm(self.loc32(loc31_out)))

        # Fourth upsample module
        up4_out = tf.nn.leaky_relu(self.up4_norm(self.up4(self.up4_upsample(loc32_out))))

        # Last convolution
        up4_concat = tf.keras.layers.concatenate([up4_out, layer1_out], 4)
        conv6_out = tf.nn.leaky_relu(self.conv6_norm(self.conv6(up4_concat)))

        # Segmentation layer and softmax 1
        logits1 = tf.keras.activations.softmax(self.seg1(loc22_out), axis=[1, 2, 3])

        # Segmentation layer and softmax 1
        logits2 = tf.keras.activations.softmax(self.seg2(loc32_out), axis=[1, 2, 3])

        # Segmentation layer and softmax 1
        logits3 = tf.keras.activations.softmax(self.seg3(conv6_out), axis=[1, 2, 3])

        return logits1, logits2, logits3

    def loss(self, y_pred, y_true):
        """
        Takes in labels and logits and returns loss
        :param y_true: labels, tensor of 1's and 0's
        :param y_pred: logits, tensor of probabilities
        :return:
        """

        y_true = tf.cast(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), float)
        y_pred = tf.cast(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), float)

        def dice_loss(y_true, y_pred):
            numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=1)
            denominator = tf.reduce_sum(y_true + y_pred, axis=1)

            return 1 - numerator / denominator

        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, folder, manager, start):
    """
    This trains the model using the data from the given folder and the given model
    :param model:
    :param folder:
    :return:
    """

    # Get the date and time in a string
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H.%M")

    # Get the list of folders in patients
    patients = sorted(os.listdir(folder))

    # Find the last index given the patients and batch_size
    last_index = int(len(patients)/model.batch_size)

    # Loop through every batch
    for i in range(start, last_index):

        print('Train Batch: ', i + 1, 'out of ', last_index)
        start_time = time.time()

        # These store the inputs and labels of the batch
        inputs = []
        labels1 = []
        labels2 = []
        labels3 = []

        print('Loading inputs...')

        # Load the patient data and turn into cubes
        for j in range(model.batch_size):

            # Lookup the patient
            patient = patients[(model.batch_size * i) + j]

            # Load their scans
            ct = np.load(folder + '/' + patient + '/CT.npy').astype(float)
            pet = np.load(folder + '/' + patient + '/PET.npy').astype(float)
            mask = np.load(folder + '/' + patient + '/mask_original.npy').astype(float)

            # Cut the random cubes
            ct_final, pet_final, mask_final, half_mask, quarter_mask = cut_random_cubes(ct, pet, mask)
            print('CT shape: ', ct_final.shape, ' || PET shape: ', pet_final.shape, ' || Mask shape: ', mask_final.shape)

            # Normalize the pet scan
            pet_final = 255.0 * pet_final/np.sum(pet_final)

            # Put the ct and pet into channels and append to the outside list
            inputs.append(np.transpose([255.0 * zero_center(normalize(ct_final)), pet_final], [1, 2, 3, 0]))

            # Put the labels into the appropriate list
            labels1.append(mask_final)
            labels2.append(half_mask)
            labels3.append(quarter_mask)

        # Turn inputs and labels into np arrays
        inputs = np.array(inputs)
        labels1 = np.array(labels1)
        labels2 = np.array(labels2)
        labels3 = np.array(labels3)

        print('Loading inputs complete')
        print('Calling and generating loss...')

        # Generate the loss
        with tf.GradientTape() as tape:
            # print(inputs.shape)
            # print(labels.shape)
            logits1, logits2, logits3 = model.call(inputs)
            loss = (model.loss(logits1, labels3)/4) + (model.loss(logits2, labels2)/2) + model.loss(logits3, labels1)

        print('Generating loss complete || Loss: ', loss)
        print('Backpropagating...')

        # Backprop
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print('Backprop complete')

        # Save the model every checkpoint batches
        if (i * model.batch_size) % model.checkpoint == 0:

            manager.save()
            print('Model Saved!')

        print("--- Batch completed in %s seconds ---" % (time.time() - start_time))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this can be the average accuracy across
    all batches or the sum as long as you eventually divide it by batch_size
    """

    last_index = int(len(test_labels) / model.batch_size)
    accuracy_ours = 0
    accuracy = 0

    for i in range(last_index):
        print('Test Batch: ', i + 1, 'out of ', last_index)
        temp_train = test_inputs[i * model.batch_size: (i + 1) * model.batch_size]
        temp_labels = test_labels[i * model.batch_size: (i + 1) * model.batch_size]

        accuracy_ours += model.accuracy(model.call(temp_train, True), temp_labels)
        accuracy += model.accuracy(model.call(temp_train), temp_labels)

    return accuracy_ours / last_index


def visualize_results(ct, pet, labels, logits):
    """
    Displays a slice and compares the logits
    :param ct:
    :param pet:
    :param labels:
    :param logits:
    :return:
    """

    display_ct_pet_processed(ct, pet, labels, logits)


def tests(model):
    random = tf.random.uniform([6, 32, 32, 32, 2])
    random1 = tf.random.uniform([6, 32, 32, 32])

    array1 = tf.convert_to_tensor([[1, 0, 0], [0, 1, 0]], dtype=tf.float16)
    array2 = tf.convert_to_tensor([[.5, .25, .25], [1, 0, 0]], dtype=tf.float16)
    array3 = tf.convert_to_tensor([[.5, .25, .25], [0.25, .5, .25]], dtype=tf.float16)

    input_data = tf.random.uniform([2, 32, 32, 32, 2])
    log1, log2, log3 = model.call(input_data)

    print(tf.shape(log1))
    print(tf.shape(log2))
    print(tf.shape(log3))
    print(model.loss(array1, array1))
    print(model.loss(array1, array2))
    print(model.loss(array1, array3))
    print(model.loss(random, random))


def test_model(model, folder):

    # Load their scans
    ct = np.load(folder + '/CT.npy').astype(float)
    pet = np.load(folder + '/PET.npy').astype(float)
    mask = np.load(folder + '/mask_original.npy').astype(float)

    # Cut the random cubes
    ct_final, pet_final, mask_final, half_mask, quarter_mask = cut_random_cubes(ct, pet, mask)
    print('CT shape: ', ct_final.shape, ' || PET shape: ', pet_final.shape, ' || Mask shape: ', mask_final.shape)
    # Put the ct and pet into channels and append to the outside list
    # Normalize the pet scan
    pet_final = pet_final / np.max(pet_final)
    # Put the ct and pet into channels and append to the outside list
    inputs = np.array([np.transpose([zero_center(normalize(ct_final)), pet_final], [1, 2, 3, 0])])

    mask1, mask2, mask3 = model.call(inputs)
    print(mask3[0,4,3,3,0])

    display_ct_pet_processed(ct_final, pet_final, mask_final, mask3[0, :, :, :, 0])


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. For CS2470 students, you must train within 10 epochs.
    You should receive a final accuracy on the testing examples for cat and dog of >=70%.
    :return: None
    '''

    # Use the titan
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Initialize the model
    model = Model()

    # For saving/loading models
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # Restore latest checkpoint
    checkpoint.restore(manager.latest_checkpoint)

    # Train it
    train(model, '/media/user1/WD750/processed_data', manager, 0)
    manager.save()

    # Test it
    for i in range(1, 100):
        test_model(model, '/media/user1/WD750/processed_data/Lung-VA-00' + str(i))


if __name__ == '__main__':
    main()
