import numpy as np
import pandas as pd
import riiideducation
import saint_plus

print("Reading questions data")
questions = pd.read_csv(
        "../input/riiid-test-answer-prediction/questions.csv", 
        usecols=["part"],
        dtype = {"part": "int8"}
)

# New data are given in groups by the riiideducation API. In each group, we can
# have multiple interactions per user, all interactions belonging to the same
# task container. The model takes as input sequences of length win_size. When
# we obtain a new group, the input sequences must be built with data from that
# group and past groups. We will keep the past interactions in a data
# structure, the interaction queue. That data structure will be updated
# everytime we receive new groups. 
 
# A group contains the questions and categories but the answer correctness is
# not given yet, we have to predict it. It is only given in the following
# group. Thus, in order to update the sequences data structure with group i, we
# have to wait until we receive group i + 1, from which we retrieve the answer
# correctness information.

# A test interaction queue filled with a group 0 and the answers from group 1
test_queue_0 = {
    "5": pd.DataFrame({
        "questions": np.array([5, 7, 3], dtype = np.int16),
        "categories": np.array([1, 1, 1], dtype = np.int8),
        "answers": np.array([1, 1, 0], dtype = np.int8),
        "positions": np.array([0, 1, 2], dtype = np.int32)
    })
}

# Test group 1
test_group_1 = pd.DataFrame({
    "row_id": [0, 1, 2],
    "timestamp": [4, 8, 5],
    "user_id": [5, 5, 8],
    "content_id": [11, 12, 1],
    "content_type_id": [0, 0, 0],
    "tast_container_id": [3, 3, 1],
    "prior_group_answers_correct": ["[1, 1, 0]", np.nan, np.nan]
})

# Interaction group keeps only the data we need from group 1 
test_interaction_group_1 = pd.DataFrame({
    "user_id": np.array([5, 5, 8], dtype = np.int64),
    "content_id": np.array([11, 12, 1], dtype = np.int16),
    "part": np.array([1, 1, 1], dtype = np.int8),
})

# Test input sequences from test_seq_0 and test_group_1 if win_size == 3. The
# values for questions are taken from test_queue and test_interaction_group but
# we add 1 to them to distinguish them from the padding value 0.
test_input_seq_1 = [
    {
        "questions": np.array([8, 4, 12], dtype = np.int16),
        "categories": np.array([1, 1, 1], dtype = np.int8),
        "positions": np.array([1, 2, 3], dtype = np.int32),
        "past_answers": np.array([1, 1, 0], dtype = np.int8)
    },
    {
        "questions": np.array([8, 4, 13], dtype = np.int16),
        "categories": np.array([1, 1, 1], dtype = np.int8),
        "positions": np.array([1, 2, 3], dtype = np.int32),
        "past_answers": np.array([1, 1, 0], dtype = np.int8)
    },
    {
        "questions": np.array([2, 0, 0], dtype = np.int16),
        "categories": np.array([1, 0, 0], dtype = np.int8),
        "positions": np.array([0, 0, 0], dtype = np.int32),
        "past_answers": np.array([2, 0, 0], dtype = np.int8)
    }
]

# Interaction queue filled with groups 0 and 1 and the answers from group 2, with win_size = 3
test_queue_1 = {
    "5": pd.DataFrame({
        "questions": np.array([3, 11, 12], dtype = np.int16),
        "categories": np.array([1, 1, 1], dtype = np.int8),
        "answers": np.array([0, 1, 0], dtype = np.int8),
        "positions": np.array([2, 3, 4], dtype = np.int32)
    }),
    "8": pd.DataFrame({
        "questions": np.array([1], dtype = np.int16),
        "categories": np.array([1], dtype = np.int8),
        "answers": np.array([1], dtype = np.int8),
        "positions": np.array([0], dtype = np.int32)
    })
}

# Test group 2
test_group_2 = pd.DataFrame({
    "row_id": [3],
    "timestamp": [9],
    "user_id": [8],
    "content_id": [13],
    "content_type_id": [0],
    "tast_container_id": [2],
    "prior_group_answers_correct": ["[1, 0, 1]"]
})

def get_prior_answers(group):
    return eval(group.loc[0, "prior_group_answers_correct"])

def get_interactions(group):
    """Extract the desired columns, add the "part" column by merging with the questions DataFrame 
    filter out the lecture interactions."""
    return group \
        .loc[group.content_type_id == 0, ["user_id", "content_id"]] \
        .merge(
            questions["part"], 
            left_on="content_id", 
            right_index=True,
            sort=False,
            validate="many_to_one") \
        .astype({"user_id": np.int64, "content_id": np.int16, "part": np.int8})

assert(get_interactions(test_group_1).equals(test_interaction_group_1))

def fill_prior_answers(queue, prior_interaction_group, prior_answers, win_size = saint_plus.win_size):
    """Given the interaction queue, the prior interaction group without answers and their answers, 
    return the queue updated with the prior interaction group and their answers."""

    prior_interaction_group = prior_interaction_group.assign(answers = 
            np.array(prior_answers, dtype = np.int8))

    # Copy the queue to avoid mutating the original dict with queue[uid] = … below to mutate 
    queue = queue.copy()
    

    for uid, data in prior_interaction_group.groupby("user_id"):

        str_uid= str(uid)

        if str_uid not in queue:
            queue[str_uid] = pd.DataFrame({ 
                "questions": np.array([], dtype = np.int16),
                "categories": np.array([], dtype = np.int8), 
                "answers": np.array([], dtype = np.int8),
                "positions": np.array([], dtype = np.int32)
            })

        new_pos = queue[str_uid].positions.values[-1] + 1 \
            if queue[str_uid].shape[0] > 0 else 0

        queue[str_uid] = queue[str_uid] \
            .append(
                pd.DataFrame({
                    "questions": data.content_id.values,
                    "positions": np.array(
                        range(new_pos, new_pos + data.shape[0]), 
                        dtype = np.int32),
                    "categories": data.part.values,
                    "answers": data.answers.values}),
                ignore_index = True) \
            .tail(win_size) \
            .reset_index(drop = True)


    return queue 

assert(all([df.equals(test_queue_1[uid]) for (uid, df)
    in fill_prior_answers(test_queue_0, test_interaction_group_1, 
        get_prior_answers(test_group_2), win_size = 3).items()
    ]))


def get_input_sequence(queue, interaction_group, win_size = saint_plus.win_size):
    """Returns the input sequences ready to feed to the model."""
    seq = []
    for row in interaction_group.itertuples():
        str_uid = str(row.user_id)
        if str_uid in queue:
            user_queue = queue[str_uid]
        else:
            user_queue = pd.DataFrame({
                "questions": np.array([], dtype = np.int16),
                "categories": np.array([], dtype = np.int8),
                "positions": np.array([], dtype = np.int32),
                "answers": np.array([], dtype = np.int8)
            })
        user_queue_size = user_queue.shape[0]
        user_seq = {
            # Add 1 to the values and use 0 for padding.
            "questions": np.concatenate((
                user_queue.questions.values[user_queue_size - win_size + 1:] + 1,
                np.array([row.content_id + 1], dtype = np.int16),
                np.full(max(0, win_size - user_queue_size - 1), 0,
                    dtype = np.int16))),
            # Use 0 for padding. Categories are numbered starting at 1,
            # we don't need to add 1 here.
            "categories": np.concatenate((
                user_queue.categories.values[user_queue_size - win_size + 1:],
                np.array([row.part], dtype = np.int8),
                np.full(max(0, win_size - user_queue_size - 1), 0,
                    dtype = np.int8))),
            # Use 0 for padding
            "positions": np.concatenate((
                user_queue.positions.values[user_queue_size - win_size + 1:],
                np.array([user_queue.positions.values[-1] + 1
                    if user_queue_size > 0 else 0 ], dtype = np.int32),
                np.full(max(0, win_size - user_queue_size - 1), 0, 
                    dtype = np.int32))),
            # If there are less than win_size elements in the queue,
            # add the start token 2 to the answers at the beginning and pad with 0
            # at the end.
            "past_answers": np.concatenate((
                np.array([2] if user_queue_size < win_size else [],
                    dtype = np.int8),
                user_queue.answers.values[user_queue_size - win_size:],
                np.full(max(0, win_size - user_queue_size - 1), 0, 
                    dtype = np.int8))),
        }
        seq.append(user_seq)
    return seq

for d1, d2 in zip(
    get_input_sequence(test_queue_0, test_interaction_group_1, win_size = 3),
    test_input_seq_1):
    for k in d2:
        assert(k in d1)
        assert(np.all(d1[k] == d2[k]))

#### API
# import riiideducation

# You can only call make_env() once, so don't lose it!
env = riiideducation.make_env()

# The API will load up to 1 GB of the test set data in memory after
# initialization. The initialization step (env.iter_test()) will require
# meaningfully more memory than that; we recommend you do not load your model
# until after making that call. The API will also consume roughly 15 minutes of
# runtime for loading and serving the data.

print("Initializing test API")
iter_test = env.iter_test()

print("Loading the model")
saint_plus.model.load_weights("../input/saint-plus-train/saved_model")

queue = {}
prior_interaction_group = None

for (test_df, sample_prediction_df) in iter_test:
    interaction_group = get_interactions(test_df)

    if test_df.group_num[0] > 0:
        prior_answers = get_prior_answers(test_df)
        queue = fill_prior_answers(queue, pior_interaction_group, prior_answers)

    sequences = get_input_sequences(queue, interaction_group)

    predictions = saint_plus.model.predict(
        (({k: np.expand_dims(d[k], axis=0) for k in d},) for d in sequences))
    
    # Select the last prediction before padding. If the sequences is shorter than win_size,
    # it is filled with the padding value 0 on the right. In the "questions" sequence, the padding value
    # is distinct from the question ids. We can thus find the position of the last "true" prediction by
    # counting the padding values.
    prediction_index = [np.sum(s["questions"] > 0) - 1 for s in sequences]

    predictions = [p[i] for i, p in zip(prediction_index, predictions)]

    test_df.loc[test_df['content_type_id'] == 0, 'answered_correctly'] = predictions 
    # The lecture rows in test_df should not be submitted.
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

    prior_interaction_group = interaction_group

        

