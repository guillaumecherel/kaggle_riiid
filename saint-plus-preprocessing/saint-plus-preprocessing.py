# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
import sys

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
#_ You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

print("Reading train data…")
train = pd.read_feather(
        "../input/riiid-train-data-multiple-formats/riiid_train.feather",
        columns=["user_id", "content_id", "content_type_id", "answered_correctly"]
)

print("Reading questions data…")
questions = pd.read_csv(
        "../input/riiid-test-answer-prediction/questions.csv", 
        usecols=["part"],
        dtype = {"part": "int8"}
)

select_questions = train.content_type_id == 0

print("Preparing interactions…")
interactions = train.loc[select_questions, ["user_id", "content_id", "answered_correctly"]]
interactions = interactions.merge(
        questions["part"], 
        left_on="content_id", 
        right_index=True,
        sort=False,
        validate="many_to_one")

# The length of the input series is the maximum number of interactions by user. 
data_dim = interactions.user_id.value_counts().max()

# def sequences(inters):
#     for (uid, data) in inters.groupby("user_id"):
#         cur_dim = data.shape[0]
#         question_sequence = np.append(
#                 data.content_id.values,
#                 np.full((data_dim - cur_dim,), -1, dtype = data.content_id.dtype))
#         category_sequence = np.append(
#                 data.part.values,
#                 np.full((data_dim - cur_dim,), -1, dtype = "int8"))
#         position_sequence = np.append(
#                 np.asarray(data.index.values, dtype="int32"),
#                 np.full((data_dim - cur_dim,), -1, dtype = "int32"))
#         answer_sequence = np.concatenate([
#                 np.array([-2], dtype = data.answered_correctly.dtype), # the "start token"
#                 data.answered_correctly.values[:-1],
#                 np.full((data_dim - cur_dim,), -1, dtype = data.answered_correctly.dtype)])
#         expected_sequence = np.append(
#                 data.answered_correctly.values,
#                 np.full((data_dim - cur_dim,), -1, dtype = data.answered_correctly.dtype))
#         yield np.array("question_sequence": question_sequence,
#                "category_sequence": category_sequence,
#                "position_sequence": position_sequence,
#                "answer_sequence": answer_sequence,
#                "expected_sequence": expected_sequence}

def sequences(inters):
    for (uid, data) in inters.groupby("user_id"):
        yield (uid, pd.DataFrame({
            "questions": data.content_id.values,
            "categories": data.part.values,
            "past_answers": np.append(
                np.array([2], dtype = data.answered_correctly.dtype), # the "start token"
                data.answered_correctly.values[:-1]),
            "answers": data.answered_correctly.values})
        )

# Clean previous output if present
for dirname, _, filenames in os.walk('riiid-train-data-sequences'):
    print("Deleting all files in " + dirname)
    for filename in filenames:
        os.remove(os.path.join(dirname, filename))

os.makedirs("riiid-train-data-sequences", exist_ok=True)

def write_to_file_numpy(seq_list, path):
    # df.reset_index(drop=True).to_feather("riiid-train-data-sequences/" + file_index + ".feather")
    que = np.empty((len(seq_list), data_dim), 
            dtype = seq_list[0]["question_sequence"].dtype)
    cat = np.empty((len(seq_list), data_dim),
            dtype = seq_list[0]["category_sequence"].dtype)
    pos = np.empty((len(seq_list), data_dim),
            dtype = seq_list[0]["position_sequence"].dtype)
    ans = np.empty((len(seq_list), data_dim),
            dtype = seq_list[0]["answer_sequence"].dtype)
    exp = np.empty((len(seq_list), data_dim),
            dtype = seq_list[0]["expected_sequence"].dtype)
    for (i, seq) in enumerate(seq_list):
        que[i,:] = seq["question_sequence"]
        cat[i,:] = seq["category_sequence"]
        pos[i,:] = seq["position_sequence"]
        ans[i,:] = seq["answer_sequence"]
        exp[i,:] = seq["expected_sequence"]

    np.savez_compressed(path + ".npz", question_sequence = que, category_sequence = cat,
        position_sequence = pos, answer_sequence = ans, 
        expected_sequence = exp)

def write_to_file_tfrecord(seq_list, path):
    # df.reset_index(drop=True).to_feather("riiid-train-data-sequences/" + file_index + ".feather")
    examples = []
    n_examples = len(seq_list)

    print("Preparing features for {} examples ".format(n_examples), end='')
    sys.stdout.flush()

    for i, seq in enumerate(seq_list):

        print(".", end='')
        sys.stdout.flush()

        feature = {
            "question_sequence": tf.train.Feature(int64_list = 
                tf.train.Int64List(value = seq["question_sequence"])),
            "category_sequence": tf.train.Feature(int64_list = 
                tf.train.Int64List(value = seq["category_sequence"])),
            "position_sequence": tf.train.Feature(int64_list = 
                tf.train.Int64List(value = seq["position_sequence"])),
            "answer_sequence": tf.train.Feature(int64_list = 
                tf.train.Int64List(value = seq["answer_sequence"])),
            "expected_sequence": tf.train.Feature(int64_list = 
                tf.train.Int64List(value = seq["expected_sequence"]))
        }

        examples.append(
            tf.train.Example(features = tf.train.Features(feature = \
            feature)).SerializeToString()
        )
    print(" Done.")
    sys.stdout.flush()

    fullpath = path + ".tfrecord"
    print("Writing to " + fullpath)
    sys.stdout.flush()

    with tf.io.TFRecordWriter(fullpath, tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
        for ex in examples:
            writer.write(ex)

    print(" Done.")
    sys.stdout.flush()


    # In [52]: b = tf.train.Example(features=tf.train.Features(feature = {"x": tf.train.Feature(int64_list
    # ...: =tf.train.Int64List(value=[1,3]))})).SerializeToString()
    # ...:
    # ...: with tf.io.TFRecordWriter("test.tfrecord") as fw:
    # ...:     fw.write(b)
    # ...:


    #     In [51]: a = tf.data.TFRecordDataset(["test.tfrecord"])
    # ...:
    # ...: for r in a:
    # ...:     print(tf.io.parse_single_example(r, {"x": tf.io.FixedLenSequenceFeature([], dtype=tf.in
    # ...: t64, allow_missing=True)}))

users_per_file = 10000

n_users = 0
n_interactions = 0
n_unique_questions = 0
n_unique_categories = 0

# Let's create an index of users and filenames by interaction to query the
# database efficiently later. This index stores the lowest interaction id and
# number of interaction for each user, as well as the file where to find that
# user. This will let us quickly find which file to open and user to look for to
# load when looking for a particular interaction.
index_filenames = []
index_users = []
index_lowest_interaction_id = []
index_nb_interactions = []

batch = []

def write_batch(batch, db_path):
    print("Writing batch to " + db_path + "…")
    sys.stdout.flush()

    with pd.HDFStore(db_path) as store:
        for i, (db_uid_batch, df_batch) in enumerate(batch):
            store[db_uid_batch] = df_batch

            if i % 100 == 0:
                print(".", end="")
            if i % 5000 == 5000 - 1:
                print("")
            sys.stdout.flush()

for (i, (uid, df)) in enumerate(sequences(interactions)):
    db_num = i // users_per_file
    db_filename = "{:05d}.h5".format(db_num)
    db_path = "riiid-train-data-sequences/{}".format(db_filename)
    db_uid = "user_" + str(uid)

    past_n_interactions = n_interactions
    cur_n_interactions = df.shape[0]

    n_users += 1 
    n_interactions += cur_n_interactions 
    # Question ids start at 0, the number of unique questions is thus the
    # maximum question id + 1.
    n_unique_questions = max(n_unique_questions, df["questions"].max()) + 1
    # Categories are numbered from 1 to 7.
    n_unique_categories = max(n_unique_categories, df["categories"].max())

    index_filenames.append(db_filename)
    index_users.append(db_uid)
    index_lowest_interaction_id.append(past_n_interactions)
    index_nb_interactions.append(cur_n_interactions)

    batch.append((db_uid, df))

    # Write to stores in batches of users_per_file users
    if i % users_per_file == users_per_file - 1:
        write_batch(batch, db_path)
        batch = []

# Write any pending interaction
write_batch(batch, db_path)


index = pd.DataFrame({
    "filename": np.array(index_filenames),
    "user": np.array(index_users), 
    "lowest_interaction_id": np.array(index_lowest_interaction_id, dtype="int32"),
    "nb_interactions": np.array(index_nb_interactions, dtype="int32")
})

with pd.HDFStore("riiid-train-data-sequences/index.h5") as store:
    store["index"] = index

print("Total number of users: " + str(n_users))
print("Total number of interactions: " + str(n_interactions))
print("Total number of unique questions: " + str(n_unique_questions))
print("Total number of unique categories: " + str(n_unique_categories))

with pd.HDFStore("riiid-train-data-sequences/meta.h5") as store:
    store["meta"] = pd.DataFrame({
        "n_users": [n_users],
        "n_interactions": [n_interactions],
        "n_unique_questions": [n_unique_questions],
        "n_unique_categories": [n_unique_categories]
    })
