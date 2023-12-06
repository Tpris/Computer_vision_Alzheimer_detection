import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Input, Activation, Add, Conv3D, MaxPooling3D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model


class Biclassifier3D():

    def __init__(self, input_shape, n_classes=2, n_filters=8, kernel_size=3, activation='relu', dropout=0.2):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout
    
    def build_model(self):
        input = Input(shape=self.input_shape)
        x = input
        x = Conv3D(self.n_filters, self.kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.n_filters, self.kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = Activation(self.activation)(x)
        #maxpooling
        x = MaxPooling3D()(x)
        #dropout
        x = Dropout(self.dropout)(x)
        #2nd conv layer
        x = Conv3D(self.n_filters*2, self.kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.n_filters*2, self.kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = Activation(self.activation)(x)
        #maxpooling
        x = MaxPooling3D()(x)
        #dense layer
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        #output layer
        output = Dense(self.n_classes, activation='softmax')(x)
        model = Model(inputs=input, outputs=output)
        return model


if __name__ == '__main__':
    input_shape = (40, 40, 40, 1)
    model = Biclassifier3D(input_shape, n_filters=8).build_model()
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)



        
