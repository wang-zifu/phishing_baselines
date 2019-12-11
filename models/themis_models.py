import tensorflow as tf
from keras import backend as K
from keras.layers import Embedding, merge, Dense, Input, Flatten, Dropout, Concatenate, LSTM, Reshape, Activation, RepeatVector, Permute, Lambda
from keras import initializers, optimizers
from keras.models import Model


def build_simple_themis(vocab, seq_max_len, w_embed_matrix=None):
    hidden_dims = 512
    EMBEDDING_DIM = 50
    w_embed = w_embed_matrix

    bw_embedding_layer = Embedding(len(vocab) + 1,
                                    EMBEDDING_DIM,
                                    weights=w_embed,
                                    input_length=seq_max_len, trainable=True,
                                    embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                      seed=None))
    bw_sequence_input = Input(shape=(seq_max_len,), name="bw_sequence_input")
    bw_embedded_sequences = bw_embedding_layer(bw_sequence_input)
    
    bw_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(bw_embedded_sequences)
    bw_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(bw_embedded_sequences)
    bw_z_concat = merge([bw_z_pos, bw_embedded_sequences, bw_z_neg], mode='concat', concat_axis=-1)

    bw_z = Dense(512, activation='tanh')(bw_z_concat)
    bw_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(bw_z)

    # ------att----------
    reshaped = Reshape((2, 512 * 2))(bw_pool_rnn)

    attention = Dense(1, activation='tanh')(reshaped)
    attention = Flatten()(attention)
    attention = Activation('sigmoid')(attention)
    attention = RepeatVector(512 * 2)(attention)
    attention = Permute([2, 1])(attention)

    sent_representation = merge([reshaped, attention], mode='mul')
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(512 * 2,))(sent_representation)

    model_final_ = Dense(512, activation='relu')(sent_representation)
    model_final_ = Dropout(0.5)(model_final_)
    model_final = Dense(1, activation='sigmoid')(model_final_)

    model = Model(input=bw_sequence_input,
                        outputs=model_final)
    adam = optimizers.adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',
                        optimizer=adam,
                        metrics=['binary_accuracy'])
    print(model.summary())