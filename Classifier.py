import tensorflow as tf
import tensorflow_addons as tfa

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.batch_size = 2
        self.checkpoint = 1
        self.hidden_size_dem = 300
        self.hidden_size_com = 500

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

        # Dense layer processing patient demographic data
        self.dense1 = tf.keras.layers.Dense(self.hidden_size_dem)
        self.dense2 = tf.keras.layers.Dense(self.hidden_size_dem)
        self.dense3 = tf.keras.layers.Dense(self.hidden_size_dem)

        # Dense layer processing demo and image data
        self.dense4 = tf.keras.layers.Dense(self.hidden_size_com)
        self.dense5 = tf.keras.layers.Dense(self.hidden_size_com)
        self.dense6 = tf.keras.layers.Dense(2)

    def call(self, image, demo):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """

        # Top layer
        conv1_out = tf.nn.leaky_relu(self.conv1_norm(self.conv1(image)))

        res11_out = tf.nn.leaky_relu(self.res11_norm(self.res11(conv1_out)))
        layer1_out = tf.nn.leaky_relu(self.res12_norm(self.res12(res11_out)) + conv1_out)

        # Second layer
        conv2_out = tf.nn.leaky_relu(self.conv2_norm(self.conv2(layer1_out)))

        res21_out = tf.nn.leaky_relu(self.res21_norm(self.res21(conv2_out)))
        layer2_out = tf.nn.leaky_relu(self.res22_norm(self.res22(res21_out)) + conv2_out)

        # Third layer
        conv3_out = tf.nn.leaky_relu(self.conv3_norm(self.conv3(layer2_out)))

        res31_out = tf.nn.leaky_relu(self.res31_norm(self.res31(conv3_out)))
        layer3_out = tf.nn.leaky_relu(self.res32_norm(self.res32(res31_out)) + conv3_out)

        # Fourth layer
        conv4_out = tf.nn.leaky_relu(self.conv4_norm(self.conv4(layer3_out)))

        res41_out = tf.nn.leaky_relu(self.res41_norm(self.res41(conv4_out)))
        layer4_out = tf.nn.leaky_relu(self.res42_norm(self.res42(res41_out)) + conv4_out)

        # Fifth layer
        conv5_out = tf.nn.leaky_relu(self.conv5_norm(self.conv5(layer4_out)))

        res51_out = tf.nn.leaky_relu(self.res51_norm(self.res51(conv5_out)))
        layer5_out = tf.nn.leaky_relu(self.res52_norm(self.res52(res51_out)) + conv5_out)

        # Demographic data dense layers
        dense1_out = tf.nn.leaky_relu(self.dense1(demo))
        dense2_out = tf.nn.leaky_relu(self.dense2(dense1_out))
        dense3_out = tf.nn.leaky_relu(self.dense3(dense2_out))

        # Combine image and patient data
        com_input = tf.concat([tf.reshape(layer5_out, [layer5_out.shape[0], -1]), dense3_out], 1)

        # Feed through other dense layers
        dense4_out = tf.nn.leaky_relu(self.dense4(com_input))
        dense5_out = tf.nn.leaky_relu(self.dense5(dense4_out))
        logits = tf.nn.softmax(self.dense6(dense5_out))

        return logits

    def loss(self, y_pred, y_true):
        """
        Takes in labels and logits and returns loss
        :param y_true: labels, one-hot tensor of 1's and 0's (batch, 2)
        :param y_pred: logits, tensor of probabilities (batch, 2)
        :return:
        """

        return tf.reduce_mean(tf.losses.binary_crossentropy(y_true, y_pred))

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


