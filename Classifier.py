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






        return logits1, logits2, logits3

    def loss(self, y_pred, y_true):
        """
        Takes in labels and logits and returns loss
        :param y_true: labels, tensor of 1's and 0's (batch, x, y, z)
        :param y_pred: logits, tensor of probabilities (batch, x, y, z, class)
        :return:
        """
        # Get tumor and background labels
        tumor_labels = y_true > 0.5
        background_labels = y_true < 0.5

        # Get tumor and background logits
        tumor_logits = y_pred[:, :, :, :, 0]
        background_logits = y_pred[:, :, :, :, 1]

        # Convert to (batch, -1) size tensors
        tumor_labels = tf.cast(tf.reshape(tumor_labels, [tf.shape(tumor_labels)[0], -1]), float)
        background_labels = tf.cast(tf.reshape(background_labels, [tf.shape(background_labels)[0], -1]), float)

        tumor_logits = tf.cast(tf.reshape(tumor_logits, [tf.shape(tumor_logits)[0], -1]), float)
        background_logits = tf.cast(tf.reshape(background_logits, [tf.shape(background_logits)[0], -1]), float)

        # Calculate softdice loss
        numerator_tumor = 2 * tf.reduce_sum(tumor_labels * tumor_logits, axis=1)
        denominator_tumor = tf.reduce_sum(tumor_labels + tumor_logits, axis=1)

        numerator_background = 2 * tf.reduce_sum(background_labels * background_logits, axis=1)
        denominator_background = tf.reduce_sum(background_labels + background_logits, axis=1)

        softdice_tumor_loss = tf.reduce_mean(1 - numerator_tumor / denominator_tumor)
        softdice_background_loss = tf.reduce_mean(1 - numerator_background / denominator_background)

        # Calculate weighted binary cross-entropy
        tumor_voxels = tf.reduce_sum(tumor_labels)
        background_voxels = tf.reduce_sum(background_labels)

        crossentropy_loss_tumor_pre = tumor_labels * tf.math.log(tumor_logits) + background_labels * tf.math.log(1 - tumor_logits)

        crossentropy_loss_tumor = (tf.reduce_sum(crossentropy_loss_tumor_pre * tumor_labels)/tumor_voxels)
        crossentropy_loss_background = (tf.reduce_sum(crossentropy_loss_tumor_pre * background_labels)/background_voxels)

        return -(crossentropy_loss_background + crossentropy_loss_tumor) + softdice_tumor_loss



    def accuracy(self, y_pred, y_true):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        # Get tumor and background labels
        tumor_labels = y_true >= 0.5


        # Get tumor and background logits
        tumor_logits = y_pred[:, :, :, :, 0]

        # Convert to (batch, -1) size tensors
        tumor_labels = tf.cast(tf.reshape(tumor_labels, [tf.shape(tumor_labels)[0], -1]), float)
        tumor_logits = tf.cast(tf.reshape(tumor_logits, [tf.shape(tumor_logits)[0], -1]), float)

        # Calculate softdice loss
        numerator_tumor = 2 * tf.reduce_sum(tumor_labels * tumor_logits, axis=1)
        denominator_tumor = tf.reduce_sum(tumor_labels + tumor_logits, axis=1)

        softdice_tumor = tf.reduce_mean(numerator_tumor / denominator_tumor)

        return softdice_tumor


