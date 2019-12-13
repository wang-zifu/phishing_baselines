import tensorflow as tf
from keras import backend as K
from keras.layers import Embedding, concatenate, Multiply, Dense, Input, Flatten, Dropout, Concatenate, LSTM, Reshape, Activation, RepeatVector, Permute, Lambda
from keras import initializers, optimizers
from keras.models import Model


def build_simple_themis(vocab, seq_max_len, embedding_dim, w_embed_matrix=None):
    hidden_dims = 512
    EMBEDDING_DIM = embedding_dim
    w_embed = w_embed_matrix

    bw_embedding_layer = Embedding(len(vocab),
                                    EMBEDDING_DIM,
                                    weights=w_embed,
                                    input_length=seq_max_len, trainable=True,
                                    embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                      seed=None))
    bw_sequence_input = Input(shape=(seq_max_len,), name="bw_sequence_input")
    bw_embedded_sequences = bw_embedding_layer(bw_sequence_input)
    
    bw_z_pos = LSTM(128, implementation=2, return_sequences=True, go_backwards=False)(bw_embedded_sequences)
    bw_z_neg = LSTM(128, implementation=2, return_sequences=True, go_backwards=True)(bw_embedded_sequences)
    bw_z_concat = concatenate([bw_z_pos, bw_embedded_sequences, bw_z_neg], axis=-1)

    bw_z = Dense(128, activation='tanh')(bw_z_concat)
    bw_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(bw_z)

    # ------att----------
    reshaped = Reshape((2, 32 * 2))(bw_pool_rnn)

    attention = Dense(1, activation='tanh')(reshaped)
    attention = Flatten()(attention)
    attention = Activation('sigmoid')(attention)
    attention = RepeatVector(32 * 2)(attention)
    attention = Permute([2, 1])(attention)

    sent_representation = Multiply()([reshaped, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(32 * 2,))(sent_representation)

    # ------model output----------
    model_final_ = Dense(32, activation='relu')(sent_representation)
    model_final_ = Dropout(0.5)(model_final_)
    model_final = Dense(1, activation='sigmoid')(model_final_)

    model = Model(input=bw_sequence_input,
                        outputs=model_final)
    adam = optimizers.adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',
                        optimizer=adam,
                        metrics=['binary_accuracy'])
    print(model.summary())
    return model


def build_simple_themis_word_char(vocab, char_vocab, seq_max_len, char_seq_max_len, embedding_dim, w_embed_matrix=None):
    hidden_dims = 512
    EMBEDDING_DIM = embedding_dim
    w_embed = w_embed_matrix

    # ------char----------
    with tf.device('/gpu:%d' % (0)):
        bc_embedding_layer = Embedding(len(char_vocab) + 1,
                                        EMBEDDING_DIM,
                                        # weights=[self.c_embed],
                                        input_length=char_seq_max_len, trainable=True,
                                        embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                          seed=None))
        bc_sequence_input = Input(shape=(char_seq_max_len,), name="bodychar_input")
        bc_embedded_sequences = bc_embedding_layer(bc_sequence_input)
        bc_z_pos = LSTM(128, implementation=2, return_sequences=True, go_backwards=False)(bc_embedded_sequences)
        bc_z_neg = LSTM(128, implementation=2, return_sequences=True, go_backwards=True)(bc_embedded_sequences)
        bc_z_concat = concatenate([bc_z_pos, bc_embedded_sequences, bc_z_neg])
        bc_z = Dense(128, activation='tanh')(bc_z_concat)
        bc_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(bc_z)
    
    # ------word----------
    with tf.device('/gpu:%d' % (3)):
        bw_embedding_layer = Embedding(len(vocab),
                                        EMBEDDING_DIM,
                                        weights=w_embed,
                                        input_length=seq_max_len, trainable=True,
                                        embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                          seed=None))
        bw_sequence_input = Input(shape=(seq_max_len,), name="bw_sequence_input")
        bw_embedded_sequences = bw_embedding_layer(bw_sequence_input)
        
        bw_z_pos = LSTM(128, implementation=2, return_sequences=True, go_backwards=False)(bw_embedded_sequences)
        bw_z_neg = LSTM(128, implementation=2, return_sequences=True, go_backwards=True)(bw_embedded_sequences)
        bw_z_concat = concatenate([bw_z_pos, bw_embedded_sequences, bw_z_neg], axis=-1)

        bw_z = Dense(128, activation='tanh')(bw_z_concat)
        bw_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(bw_z)

    # ------att---------- should be concat of word and char as input
    word_char_concat = concatenate([bw_pool_rnn, bc_pool_rnn])
    reshaped = Reshape((2, 64 * 2))(word_char_concat)

    attention = Dense(1, activation='tanh')(reshaped)
    attention = Flatten()(attention)
    attention = Activation('sigmoid')(attention)
    attention = RepeatVector(64 * 2)(attention)
    attention = Permute([2, 1])(attention)

    sent_representation = Multiply()([reshaped, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(64 * 2,))(sent_representation)

    # ------model output----------
    model_final_ = Dense(64, activation='relu')(sent_representation)
    model_final_ = Dropout(0.5)(model_final_)
    model_final = Dense(1, activation='sigmoid')(model_final_)

    model = Model(input=[bw_sequence_input, bc_sequence_input],
                        outputs=model_final)
    adam = optimizers.adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',
                        optimizer=adam,
                        metrics=['binary_accuracy'])
    print(model.summary())
    return model
