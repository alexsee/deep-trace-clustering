import tensorflow as tf
import numpy as np

class AutoencoderRepresentation(object):
    def __init__(self,
                event_log):
        self._event_log = event_log
    
        # properties
        self._model = None
        self._encoder = None
        
        self._event_attribute_encodes = None
        self._event_attribute_encoders = None
        self._case_attribute_encodes = None
        self._case_attribute_encoders = None
        
    
    def build_model(self, input_dim, encoder_dim=2):
        # encode event and case attributes
        self._event_attribute_encodes, self._event_attribute_encoders = self._event_log.encode_event_attributes()
        self._case_attribute_encodes, self._case_attribute_encoders = self._event_log.encode_case_attributes()
        
        # build model
        input_seq = tf.keras.layers.Input(shape=(input_dim,), name='input')
        encoded = input_seq
        
        # representation
        encoded = tf.keras.layers.Dense(int(input_dim * 0.2), activation=tf.nn.relu)(encoded)
        encoded = tf.keras.layers.Dropout(0.5)(encoded)
        encoded = tf.keras.layers.Dense(encoder_dim, activation=tf.nn.relu)(encoded)
        
        decoded = tf.keras.layers.Dropout(0.5)(encoded)
        decoded = tf.keras.layers.Dense(int(input_dim * 0.2), activation=tf.nn.relu)(decoded)
        decoded = tf.keras.layers.Dropout(0.5)(decoded)
        decoded = tf.keras.layers.Dense(int(input_dim), activation=tf.nn.sigmoid, name='output')(decoded)

        # the autoencoder model
        self._model = tf.keras.models.Model(input_seq, decoded)

        # encoder model
        self._encoder = tf.keras.models.Model(input_seq, encoded)

        # configure model
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_2=0.99),
            loss='mse'
        )
        
    
    def model_summary(self):
        self._model.summary()
        
    
    def fit(self, epochs=50, batch_size=256, verbose=0):
        sequences = self._event_log.event_attributes_flat_onehot_features_2d
        
        self._model.fit(sequences, sequences, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    
    def predict(self):
        return self._encoder(self._event_log.event_attributes_flat_onehot_features_2d)