from __future__ import absolute_import

import tensorflow as tf
from preprocess import *
import time
from datetime import datetime
import math
import Segmentor
import matplotlib.pyplot as plt
import Classifier


def train_seg(model, folder, manager, start, save_dir):
    """
    This trains the model using the data from the given folder and the given model
    :param model:
    :param folder:
    :return:
    """

    # Get the list of folders in patients
    patients = sorted(os.listdir(folder))

    # Find the last index given the patients and batch_size
    last_index = int(len(patients)/model.batch_size)

    # Keeps track of how many we've skipped
    added_index = 0

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

        try:
            # Load the patient data and turn into cubes
            j = 0
            while j < model.batch_size:

                # Lookup the patient
                patient = patients[(model.batch_size * i) + j + added_index]
                print('Current patient: ', patient)

                # Load their scans
                ct = np.load(folder + '/' + patient + '/CT.npy').astype(float)
                pet = np.load(folder + '/' + patient + '/PET.npy').astype(float)
                mask = np.load(folder + '/' + patient + '/mask_original.npy').astype(float)

                # Cut the random cubes
                ct_final, pet_final, mask_final, half_mask, quarter_mask, nodule = cut_random_cubes(ct, pet, mask)

                if nodule:
                    # Put the ct and pet into channels and append to the outside list
                    inputs.append(np.transpose([(normalize(ct_final)), normalize_pet(pet_final)], [1, 2, 3, 0]))

                    # Put the labels into the appropriate list
                    labels1.append(mask_final)
                    labels2.append(half_mask)
                    labels3.append(quarter_mask)
                    j += 1
                else:
                    added_index += 1
                    continue

            # Turn inputs and labels into np arrays
            inputs = np.array(inputs)
            labels1 = np.array(labels1)
            labels2 = np.array(labels2)
            labels3 = np.array(labels3)

            print('Input shape: ', inputs.shape)

        except:
            print('Error loading patient data!')
            continue


        print('Loading inputs complete')
        print('Calling and generating loss...')

        # Generate the loss
        with tf.GradientTape() as tape:
            # print(inputs.shape)
            # print(labels.shape)
            logits1, logits2, logits3 = model.call(inputs)
            loss = (model.loss(logits1, labels3)/4) + (model.loss(logits2, labels2)/2) + model.loss(logits3, labels1)

            # print accuracy
            print('Accuracy: ', model.accuracy(logits3, labels1).numpy())

        print('Loss: ', loss.numpy())

        # Make sure loss is not nan
        if not math.isnan(loss):

            print('Backpropagating...')

            # Backprop
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print('Backprop complete')

            # Save the model every checkpoint batches
            if (i * model.batch_size) % model.checkpoint == 0:

                # Save the model
                manager.save()
                print('Model Saved!')

                # Save the loss
                try:
                    loss_list = np.load(save_dir + '/loss.npy').tolist()
                except:
                    print('Created Loss list')
                    loss_list = []
                loss_list.append(loss.numpy())
                print(loss_list[-10:])
                np.save(save_dir + '/loss', loss_list)

                # Test it
                display_ct_pet_processed(inputs, labels1, logits3)
            print("Patients skipped: ", added_index)
            print("--- Batch completed in %s seconds ---" % (time.time() - start_time))
        else:
            print('Skipping backprop, loss is invalid')
            continue


def train_class(model, folder, demo_data, manager, start, save_dir):
    """
    This trains the model using the data from the given folder and the given model
    :param model:
    :param folder:
    :return:
    """

    # Get the list of folders in patients
    patients = sorted(os.listdir(folder))

    # Find the last index given the patients and batch_size
    last_index = int(len(patients)/model.batch_size)

    # Keeps track of how many we've skipped
    added_index = 0

    # Loop through every batch
    for i in range(start, last_index):

        print('Train Batch: ', i + 1, 'out of ', last_index)
        start_time = time.time()

        # These store the inputs and labels of the batch
        image_inputs = []
        demo_inputs = []
        labels = []

        print('Loading inputs...')

        try:
            # Load the patient data and turn into cubes
            j = 0
            while j < model.batch_size:

                # Lookup the patient
                patient = patients[(model.batch_size * i) + j + added_index]
                print('Current patient: ', patient)

                # Load their scans
                ct = np.load(folder + '/' + patient + '/CT.npy').astype(float)
                pet = np.load(folder + '/' + patient + '/PET.npy').astype(float)
                mask = np.load(folder + '/' + patient + '/mask_original.npy').astype(float)

                # Cut the random cubes
                ct_final, pet_final, nodule = cut_cubes_mask(ct, pet, mask)

                # If it's a nodule, add it, otherwise, move to next patient
                if nodule:

                    # Put the ct and pet into channels and append to the outside list
                    image_inputs.append(np.transpose([(normalize(ct_final)), normalize_pet(pet_final)], [1, 2, 3, 0]))

                    # Get their demographic information
                    demo_inputs.append(demo_data[patient][:6])
                    labels.append(demo_data[patient][-1])
                    j += 1

                else:

                    added_index += 1
                    continue

            # Turn inputs and labels into np arrays
            image_inputs = np.array(image_inputs)
            demo_inputs = np.array(demo_inputs)
            temp_labels = np.array(labels).astype(int) - 1
            b = np.zeros((temp_labels.size, 2))
            b[np.arange(temp_labels.size), temp_labels] = 1
            labels = b
            print('Image shape: ', image_inputs.shape)
            print('Demo shape: ', demo_inputs.shape)
            print(demo_inputs)
            print('Labels shape: ', labels.shape)
            print(labels)

        except Exception as e:
            print(e)
            print('Error loading patient data!')
            continue


        print('Loading inputs complete')
        print('Calling and generating loss...')

        # Generate the loss
        with tf.GradientTape() as tape:
            # print(inputs.shape)
            # print(labels.shape)
            logits = model.call(image_inputs, demo_inputs)
            print(logits, labels)
            loss = model.loss(logits, labels)

            # print accuracy
            print('Accuracy: ', model.accuracy(logits, labels).numpy())

        print('Loss: ', loss.numpy())

        # Make sure loss is not nan
        if not math.isnan(loss):

            print('Backpropagating...')

            # Backprop
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print('Backprop complete')

            # Save the model every checkpoint batches
            if (i * model.batch_size) % model.checkpoint == 0:

                # Save the model
                manager.save()
                print('Model Saved!')

                # Save the loss
                try:
                    loss_list = np.load(save_dir + '/loss.npy').tolist()
                except:
                    print('Created Loss list')
                    loss_list = []
                loss_list.append(loss.numpy())
                print(loss_list[-10:])
                np.save(save_dir + '/loss', loss_list)


            print("Patients skipped: ", added_index)
            print("--- Batch completed in %s seconds ---" % (time.time() - start_time))
        else:
            print('Skipping backprop, loss is invalid')
            continue


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
    print(model.loss(random, random[:, :, :, :, 0]))
    # print(model.loss(random, random1))


def test_model(model, folder):

    # Load their scans
    ct = np.load(folder + '/CT.npy').astype(float)
    pet = np.load(folder + '/PET.npy').astype(float)
    mask = np.load(folder + '/mask_original.npy').astype(float)

    # Cut the random cubes
    ct_final, pet_final, mask_final, half_mask, quarter_mask, _ = cut_random_cubes(ct, pet, mask)
    print('CT shape: ', ct_final.shape, ' || PET shape: ', pet_final.shape, ' || Mask shape: ', mask_final.shape)

    # Put the ct and pet into channels and append to the outside list
    inputs = np.array([np.transpose([(normalize(ct_final)), normalize_pet(pet_final)], [1, 2, 3, 0])])

    # Call the model
    mask1, mask2, mask3 = model.call(inputs)

    inputs = []
    labels1 = []
    labels2 = []
    labels3 = []

    # Put the labels into the appropriate list
    inputs.append(np.transpose([(normalize(ct_final)), normalize_pet(pet_final)], [1, 2, 3, 0]))
    labels1.append(mask_final)
    labels2.append(half_mask)
    labels3.append(quarter_mask)

    # Turn inputs and labels into np arrays
    inputs = np.array(inputs)
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    labels3 = np.array(labels3)

    loss = (model.loss(mask1, labels3) / 4) + (model.loss(mask2, labels2) / 2) + model.loss(mask3, labels1)
    print('Loss: ', loss)
    # Display them
    display_ct_pet_processed_test(inputs, labels1, mask3)


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

    # Segmentor or Classifier (0 or 1)
    choose = 1

    # Initialize the model
    if choose:
        model = Classifier.Model()
    else:
        model = Segmentor.Model()


    # For saving/loading models
    # Get the date and time in a string
    now = datetime.now()
    if choose:
        mod_str = 'classifier'
    else:
        mod_str = 'segmentor'
    dt_string = now.strftime("%d.%m.%Y_%H.%M") + mod_str
    checkpoint_dir = dt_string
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=80)

    print(manager.checkpoints)
    print(manager.latest_checkpoint)

    # Restore latest checkpoint
    # checkpoint.restore('./checkpoints_data/ckpt-672')
    checkpoint.restore(manager.latest_checkpoint)


    # Train it
    if choose:
        train_class(model, './processed_data', import_excel('./Lung-PET.xlsx', 'Lung-PENN'), manager, 0, checkpoint_dir)
    else:
        train_seg(model, './processed_data', manager, 0, checkpoint_dir)
        for i in range(10):
            train_seg(model, './processed_data', manager, 0, checkpoint_dir)
        manager.save()

    # Graph loss
    fig = plt.figure()
    ax = plt.axes()
    loss_graph = np.load(checkpoint_dir + '/loss.npy')
    print(len(loss_graph))
    x = list(range(len(loss_graph)))
    ax.plot(x, loss_graph)
    plt.show()

    # Test it
    if choose:
        print('hi')
    else:
        patients = sorted(os.listdir('./processed_data/'))
        for patient in patients:
            print('Current patient: ', patient)
            test_model(model, './processed_data/' + patient)


if __name__ == '__main__':
    main()
