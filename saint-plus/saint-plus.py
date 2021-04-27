# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import bisect
import gc
import math
import numpy as np # linear algebra
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
import time
import sys

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
#_ You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Input sequence length
win_size = 100
# Number of heads
h = 8
# Number of stacked encoder and decoder blocks
N = 4
# Latent space dimension. Dimension of all layer output (including embeddings).
d_model = 512
# d_model = 256
# Dimensions of key, query and value vectors
d_key = d_model // h
# d_key = 8
# Dimension of the internal layer of the feed forward networks (see function
# ffn).
dff = 2048
# dff = 512
# Dropout rate
dropout_rate = 0.1


#################
# Load the data #
#################


def windows(x, size, padding, skip):
    """Returns a 2D array where the rows are windows over the 1D array `x`. The
    first window starts with only the first element in x at its left-most
    position, other elements are filled with `padding`. The subsequent windows
    progress from left to right in the array. `skip` makes the function drop the
    `skip` first windows. For example: 

        In [84]: windows(np.array(range(10)), 5, -1, 0)
        Out[84]:
        array([[ 0, -1, -1, -1, -1],
               [ 0,  1, -1, -1, -1],
               [ 0,  1,  2, -1, -1],
               [ 0,  1,  2,  3, -1],
               [ 0,  1,  2,  3,  4],
               [ 1,  2,  3,  4,  5],
               [ 2,  3,  4,  5,  6],
               [ 3,  4,  5,  6,  7],
               [ 4,  5,  6,  7,  8],
               [ 5,  6,  7,  8,  9]])
    """
    return np.vstack([np.append( 
        x[max(0,i - size): i], 
        np.full(max(0, size - i), padding, dtype = x.dtype))
        for i in range(skip + 1, x.size + 1)])


class SequencesHDF5(keras.utils.Sequence):
    def __init__(self, path, batch_size, win_size):
        self.path = path
        self.batch_size = batch_size
        self.win_size = win_size

        with pd.HDFStore(os.path.join(self.path, "index.h5"), "r") as store:
            self.index = store["index"]
                       
        with pd.HDFStore(os.path.join(self.path, "meta.h5"), "r") as store:
            meta = store["meta"]
            self.n_users = meta["n_users"][0]
            self.n_interactions = meta["n_interactions"][0]
            # Add 1 because we count the padding value 0 as an additionnal
            # question.
            self.n_unique_questions = meta["n_unique_questions"][0] + 1
            # Categories start at 1, we use 0 as a padding value which we count
            # as an additionnal category.
            self.n_unique_categories = meta["n_unique_categories"][0] + 1

        # Add 1 to count the padding value 0.
        self.n_unique_positions = self.index.nb_interactions.max() + 1

        print("[SequencesHDF5] Total number of users: " + str(self.n_users))
        print("[SequencesHDF5] Total number of interactions: " + str(self.n_interactions))
        print("[SequencesHDF5] Total number of unique questions: " + str(self.n_unique_questions))
        print("[SequencesHDF5] Total number of unique categories: " + str(self.n_unique_categories))

    def __len__(self):
        return math.ceil(self.n_interactions / self.batch_size)

    def lookup_batch(self, batch_idx):
        """Returns the interactions to build batch idx in the form [(filename,
        [(user, start, n)])], where start and n specify the range of interactions to
        use for user (start is relative to the this user first interaction).
        (None, None, None, n) means that there is not enough data to make up a full batch
        and that we would need n additionnal interactions.
        """
        ranges = []

        nb_interactions = 0
        interaction_idx = batch_idx * self.batch_size
        user_idx = bisect.bisect_right(self.index.lowest_interaction_id, 
            interaction_idx) - 1

        while nb_interactions < self.batch_size and user_idx < self.n_users:
            filename = self.index.filename[user_idx]
            user = self.index.user[user_idx]
            start_pos = interaction_idx \
                - self.index.lowest_interaction_id[user_idx]
            remaining_interactions = self.index.nb_interactions[user_idx] \
                - start_pos 
            missing_interactions = self.batch_size - nb_interactions
            n = min(missing_interactions, remaining_interactions)

            if len(ranges) == 0 or ranges[-1][0] != filename:
                ranges.append((filename, []))
            ranges[-1][1].append((user, start_pos, n))

            nb_interactions += n
            user_idx += 1
            interaction_idx += n 

        if nb_interactions < self.batch_size:
            missing_interactions = self.batch_size - nb_interactions
            ranges.append((None, None, None, missing_interactions))

        return ranges

    def __getitem__(self, batch_idx):
        if batch_idx >= self.__len__() :
            raise IndexError("batch index out of range")

        ranges = self.lookup_batch(batch_idx)

        que = []
        cat = []
        pos = []
        pan = []
        ans = []

        for (fn, user_start_n) in ranges:
            # Make empty windows if there are not enough interactions
            # to make a full batch
            if fn == None:
                que.append(np.full((n, self.win_size), 0, dtype = np.int16))
                cat.append(np.full((n, self.win_size), 0, dtype = np.uint8))
                pos.append(np.full((n, self.win_size), 0, dtype = np.int32))
                pan.append(np.full((n, self.win_size), 0, dtype = np.uint8))
                ans.append(np.full((n, self.win_size), 0, dtype = np.uint8))
            else:
                with pd.HDFStore(os.path.join(self.path, fn), "r") as store:
                    for user, start, n in user_start_n:
                        df = store["/" + user].loc[max(0, start - self.win_size + 1) \
                            : start + n - 1, :]
                        skip = min(start, self.win_size - 1)
                        # Add 1 to the values and use 0 for padding.
                        que.append(windows(df.questions.values + 1, self.win_size, 0,
                            skip))
                        # Use 0 for padding. Categories are numbered starting at 1,
                        # we don't need to add 1 here.
                        cat.append(windows(df.categories.values, self.win_size, 0,
                            skip))
                        # Use 0 for padding.
                        pos.append(windows(
                            np.array(range(
                                max(0, start - self.win_size + 1) + 1, 
                                    start + n + 1), 
                                dtype = "int32"), 
                            self.win_size, 0, skip))
                        # For past answers and expected answers below, use 0 for padding
                        # but don't distinguish it from the answer 0. The padding values won't be
                        # used for prediction anyway.
                        pan.append(windows(df.past_answers.values, self.win_size, 0,
                            skip))
                        ans.append(windows(df.answers.values, self.win_size, 0,
                            skip))

        inputs = {  
            "questions": np.vstack(que),
            "categories": np.vstack(cat),
            "positions": np.vstack(pos),
            "past_answers": np.vstack(pan)
        }
        outputs = { "answers": np.vstack(ans) }

        return inputs, outputs
        


# sequences = SequencesHDF5("../input/saint-plus-preprocessing", 64, 100)



###################
# Build the model #
###################


def build_model(
    sequences,
    #window size
    win_size = win_size, 
    # number of heads
    h = h,
    # Number of stacked encoder and decoder blocks
    N = N,
    # Latent space dimension. Dimension of all layer output (including embeddings). 
    #d_model = 512
    d_model = d_model,
    # Dimensions of key, query and value vectors
    # d_key = d_model // h
    d_key = d_key,
    # Dimension of the internal layer of the feed forward networks (see function 
    # ffn).
    #dff = 2048
    dff = dff,
    # Dropout rate
    dropout_rate = dropout_rate,
    plot_models = False,
    # Not compiling the model can lead to faster prediction
    # https://stackoverflow.com/questions/58378374/why-does-keras-model-predict-slower-after-compile
    compile_model = True):
    question_input = keras.Input(shape=(win_size,), dtype="int16", 
            name = "questions")
    category_input = keras.Input(shape=(win_size,), dtype="uint8", 
            name = "categories")
    position_input = keras.Input(shape=(win_size,), dtype="int32", 
            name = "positions")
    past_answer_input = keras.Input(shape=(win_size,), dtype="uint8", 
            name = "past_answers")

    # Vocabulary size: maximum question id + 1 (question ids start at 0)
    # + 1 for the padding value "-1"
    question_vocabulary_size = sequences.n_unique_questions
    question_embedding = kl.Embedding(question_vocabulary_size, d_model, 
            input_length = win_size)(question_input)
    print("Question embedding", question_embedding.shape)

    # Categories start at 1, add 1 for the padding value
    category_vocabulary_size = sequences.n_unique_categories
    category_embedding = kl.Embedding(category_vocabulary_size, d_model, 
            input_length = win_size)(category_input)
    print("Category embedding", category_embedding.shape)

    # Positions start at 0, add 1 for the padding value
    position_vocabulary_size = sequences.n_unique_positions
    position_embedding = kl.Embedding(position_vocabulary_size, d_model, 
            input_length = win_size)(position_input)
    print("Position embedding", position_embedding.shape)

    # Correctness is boolean, add 1 for the start token. We use the value 0 for
    # padding, not distinguishing it fro mthe answer 0, we thus don't need to
    # extend the vocabulary size for it.
    past_answer_vocabulary_size = 3
    past_answer_embedding = kl.Embedding(past_answer_vocabulary_size, d_model, 
            input_length = win_size)(past_answer_input)
    print("Past answer embedding", past_answer_embedding.shape)

    # Exercises embedding, dim (batch, win_size, d_model)
    exercise_embedding = kl.Add()([question_embedding, category_embedding, 
        position_embedding])
    exercise_embedding = kl.Dropout(dropout_rate)(exercise_embedding)
    print("Exercise embedding", exercise_embedding.shape)

    # Response embedding, dim (batch, win_size, d_model)
    response_embedding = kl.Add()([past_answer_embedding, position_embedding])
    response_embedding = kl.Dropout(dropout_rate)(response_embedding)
    print("Response embedding", exercise_embedding.shape)

    # Mask used in the multihead attention layers. 
    # Used to set the upper triangular matrix Q_i * K_i to -Inf,
    # such that they are set to 0 after the softmax operation.
    attention_mask = kl.Lambda(
            lambda x: 
                x + tf.math.multiply_no_nan(
                    np.inf, 
                    tf.linalg.LinearOperatorLowerTriangular(
                        tf.constant(1.0, shape = (x.shape[1], x.shape[2]), dtype="float")
                    ).to_dense() - 1
                )
    )


    def head(name="head"):
        """A single attention head.

        """
        q_i = keras.Input((win_size, d_key), name = "q_i")
        k_i = keras.Input((win_size, d_key), name = "k_i")
        v_i = keras.Input((win_size, d_key), name = "v_i")

        res = kl.Dot(axes=(2,2))([q_i, k_i]) * np.sqrt(d_key)
        res = attention_mask(res)
        res = kl.Softmax()(res)
        res = kl.Dot(axes=(2,1))([res, v_i])
        return keras.Model(inputs = [q_i, k_i, v_i], outputs = [res], name = name)

    if plot_models:
        keras.utils.plot_model(head(), "graph_head.png", show_shapes = True)



    def multihead(name="multihead"):
        """Multihead attention layer with upper triangular mask.

        q, k, v: Tensors with shape (batch, win_size, d_model)

        returns: Tensor with shape (batch, win_size, d_model)."""

        q = keras.Input((win_size, d_model), name = "q")
        k = keras.Input((win_size, d_model), name = "k")
        v = keras.Input((win_size, d_model), name = "v")

        q_heads = [kl.Dense(d_key, use_bias = False)(q) for _ in range(h)]
        k_heads = [kl.Dense(d_key, use_bias = False)(k) for _ in range(h)]
        v_heads = [kl.Dense(d_key, use_bias = False)(v) for _ in range(h)]

        heads = [head(name="head_"+str(i))([q_i, k_i, v_i]) for (i,(q_i, k_i, v_i))
                in enumerate(zip(q_heads, k_heads, v_heads))]

        out = kl.Concatenate(axis=2)(heads)
        out = kl.Dense(d_model, use_bias=False)(out)
        out = kl.Dropout(dropout_rate)(out)
        
        return keras.Model(inputs = [q, k, v], outputs = [out], name = name)

    if plot_models:
        keras.utils.plot_model(multihead(), "graph_multihead.png", show_shapes = True)


    def ffn(name="ffn"):
        """Feed forward network.

        x: Tensor with shape (batch, win_size, d_model)"""
        x = keras.Input((win_size, d_model), name="x")
        out = kl.Dense(dff, activation = "relu", use_bias = True)(x)
        out = kl.Dense(d_model, use_bias=True)(out)
        out = kl.Dropout(dropout_rate)(out)
        
        return keras.Model(x, out, name = name) 

    if plot_models:
        keras.utils.plot_model(ffn(), "graph_ffn.png", show_shapes = True)



    def encoder_layer(name="encoder_layer"):
        """Encoder layer.

        x: Tensor with shape (batch, win_size, d_model)

        returns: Tensor with shape (batch, win_size, d_model)"""
        x = keras.Input((win_size, d_model), name="x")
        x_norm = kl.LayerNormalization()(x)
        m = x + multihead()([x_norm, x_norm, x_norm]) 
        out = m + ffn()(kl.LayerNormalization()(m))
        return keras.Model(x, out, name = name)

    if plot_models:
        keras.utils.plot_model(encoder_layer(), "graph_encoder_layer.png", show_shapes = True)



    def encoder_block(name = "encoder_block"):
        """Encoder block, sequence of N encoder layers.

        ee: exercise embedding sequence, Tensor of shape (batch, win_size, d_model)

        returns: Tensor of shape (batch, win_size, d_model)"""
        ee = keras.Input((win_size, d_model), name="exercise_embeding")

        out = encoder_layer(name = "encoder_layer_0")(ee)

        for i in range(1, N):
            out = encoder_layer(name = "encoder_layer_" + str(i))(out)

        return keras.Model(ee, out, name = name)

    if plot_models:
        keras.utils.plot_model(encoder_block(), "graph_encoder_block.png", show_shapes = True)



    def decoder_layer(name = "decoder_layer"):
        """Decoder layer.

        x: Tensor of shape (batch, win_size, d_model).

        o_enc: Encoder output. Tensor of shape (batch, win_size, d_model).

        returns: Tensor of shape (batch, win_size, d_model)."""
        x = keras.Input((win_size, d_model), name="x")
        o_enc = keras.Input((win_size, d_model), name="encoder_output")

        x_norm = kl.LayerNormalization()(x)
        m1 = x + multihead(name = "multihead_1")([x_norm, x_norm, x_norm])

        m1_norm = kl.LayerNormalization()(m1)
        o_enc_norm = kl.LayerNormalization()(o_enc)
        m2 = m1 + multihead(name = "multihead_2")([m1_norm, o_enc_norm, o_enc_norm])

        m2_norm = kl.LayerNormalization()(m2)
        out = m2 + ffn()(m2_norm)
        return keras.Model(inputs = [x, o_enc], outputs = [out], name = name) 

    if plot_models:
        keras.utils.plot_model(decoder_layer(), "graph_decoder_layer.png", show_shapes = True)



    def decoder_block(name = "decoder_block"):
        """Decoder block, sequence of N decoder layers.

        re: response embedding sequence. Tensor of shape (batch, win_size, d_model).

        o_enc: Encoder output. Tensor of shape (batch, win_size, d_model).

        returns: Tensor of shape (batch, win_size, d_model).
        """
        re = keras.Input((win_size, d_model), name="re")
        o_enc = keras.Input((win_size, d_model), name="encoder_output")

        out = decoder_layer(name = "decoder_layer_0")([re, o_enc])

        for i in range(1, N):
            out = decoder_layer(name = "decoder_layer_" + str(i))([out, o_enc])

        return keras.Model(inputs = [re, o_enc], outputs = [out], name = name)

    if plot_models:
        keras.utils.plot_model(decoder_block(), "graph_decoder_block.png", show_shapes = True)



    print("Building the model")
    encoder_output = encoder_block()(exercise_embedding)
    x = decoder_block()([response_embedding, encoder_output])
    x = kl.Dense(1, activation = "sigmoid", use_bias = True)(x)
    predicted_response = kl.Flatten(name = "answers")(x)

    model = keras.Model(
            inputs = [question_input, category_input, position_input, past_answer_input], 
            outputs = [predicted_response])

    if plot_models:
        keras.utils.plot_model(model, "graph_model.png", show_shapes = True)

    if compile_model:
        print("Compiling.")
        model.compile(
            optimizer=keras.optimizers.Adam() ,
            loss={"answers": keras.losses.BinaryCrossentropy()},
            metrics={"answers": [
                tf.keras.metrics.BinaryAccuracy(), 
                tf.keras.metrics.AUC()
            ]},
        )
    else:
        print("Skipping model compilation.")
    
    return model



