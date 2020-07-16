import tensorflow as tf
import numpy as np

class EmbeddingPredict(object):
    def __init__(self,
                event_log):
        self._event_log = event_log
    
        # properties
        self._model = None
        self._input_layers = None
        self._embedding_layers = None
        self._output_layers = None
        self._rnn = None

        self._event_attribute_encodes = None
        self._event_attribute_encoders = None
        self._case_attribute_encodes = None
        self._case_attribute_encoders = None
    
    
    def build_model(self, embedding_dim=32, gru_dim=32, rnn='gru', verbose=0):
        # encode event and case attributes
        self._event_attribute_encodes, self._event_attribute_encoders = self._event_log.encode_event_attributes()
        self._case_attribute_encodes, self._case_attribute_encoders = self._event_log.encode_case_attributes()
        
        # build model
        self._input_layers = []
        self._output_layers = []
        self._embedding_layers = []
        
        # build embeddings layers
        for i, event_attribute in enumerate(self._event_log.event_attributes):
            inp = tf.keras.layers.Input(shape=(None, ))
            embedding = tf.keras.layers.Embedding(input_dim=len(self._event_attribute_encoders[event_attribute].classes_) + 1, 
                                                  output_dim=embedding_dim, 
                                                  name='embedding_' + event_attribute.replace(':',''))(inp)

            self._input_layers.append(inp)
            self._embedding_layers.append(embedding)
        
        if len(self._embedding_layers) > 1:
            self._embedding_layers = tf.concat(self._embedding_layers, axis=-1)
            
        # rnn
        if rnn == 'gru':
            self._rnn = tf.keras.layers.GRU(gru_dim, return_sequences=False)(self._embedding_layers)
        else:
            self._rnn = tf.keras.layers.LSTM(gru_dim, return_sequences=False)(self._embedding_layers)
        
        # case attributes
        for case_attribute_encoder in self._case_attribute_encoders:
            fc = tf.keras.layers.Dense(len(case_attribute_encoder.classes_), activation="softmax")(self._rnn)
            self._output_layers.append(fc)
        
        # build model
        self._model = tf.keras.Model(inputs=self._input_layers, outputs=self._output_layers)
        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        
    def model_summary(self):
        self._model.summary()
    
    
    def fit(self, epochs=50, batch_size=256, verbose=0):
        # early stopping
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        
        # fit model
        self._model.fit(self._event_attribute_encodes, self._case_attribute_encodes, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[earlystop_callback])


    def predict(self):
        pred_model = tf.keras.Model(inputs=self._input_layers, outputs=[*self._output_layers, self._rnn])
        predictions = pred_model.predict(self._event_attribute_encodes)

        rnn_vector = predictions[-1]
        embedding_vector = np.hstack(predictions[0:len(predictions) - 1])

        return pred_model, rnn_vector, embedding_vector
    
    def write_vector_projectors(self, layer_num=2, encoder='concept:name'):
        import io

        weights = self._model.layers[layer_num].get_weights()[0]
        encoder = self._event_attribute_encoders[encoder]

        out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
        out_m = io.open('meta.tsv', 'w', encoding='utf-8')

        for num, word in enumerate(encoder.classes_):
            vec = weights[num] # skip 0, it's padding.
            out_m.write(word + "\n")
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_v.close()
        out_m.close()