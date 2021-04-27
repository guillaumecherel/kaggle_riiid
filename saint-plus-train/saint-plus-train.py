import time

t0 = time.time()

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import saint_plus


#######################
# Training parameters #
#######################

# Stop the training when the overall script running time reaches that many seconds
max_script_duration = 9 * 3600 - 20 * 60  # 8h45 (GPU)
# max_script_duration = 2 * 3600 + 45 * 60 # 2h45 (TPU)
# max_script_duration = 1 * 60 # 1 minute

warmup_steps = 4000
batch_size = 64 # 16 * strategy.num_replicas_in_sync
steps_per_epoch = 1000
full_dataset_passes = 40
shuffle_dataset = True

load_train_checkpoint = True


###################################
# Detect parallelization hardware #
###################################

try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError: # no TPU found, detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines

print("Number of accelerators: ", strategy.num_replicas_in_sync)



###################
# Load input data #
###################

print("Preparing sequences")

sequences = saint_plus.SequencesHDF5("/kaggle/input/saint-plus-preprocessing/riiid-train-data-sequences", batch_size, saint_plus.win_size)


###################
# Train the model #
###################



class LearningRate(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs = None):
        self.epoch = epoch
    
    def on_train_batch_begin(self, batch, logs=None):
        step = (self.epoch * steps_per_epoch) + batch + 1
        lr = saint_plus.d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
        # print("LearningRate set to {} for step {} (epoch {}, batch {})".format(lr, step, self.epoch, batch))
        keras.backend.set_value(self.model.optimizer.lr, lr)
        
    def on_epoch_end(self, epoch, logs = None):
        print("Current learning rate: {}".format(keras.backend.get_value(self.model.optimizer.lr)))
        
class MaxDuration(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        if (now - t0) > max_script_duration:
            print("Maximum duration reached, stopping the training")
            self.model.stop_training = True
            self.last_epoch = epoch



# instantiating the model in the strategy scope creates the model on the TPU
with strategy.scope():
    model = saint_plus.build_model(sequences)
    if load_train_checkpoint:
        model.load_weights("../input/saintplus-train-checkpoint/saved_model")
        with open("../input/saintplus-train-checkpoint/last_epoch", "r") as f:
            initial_epoch = int(f.read())
        import shutil
        try:
            shutil.copyfile("../input/saintplus-train-checkpoint/training_log.csv", "training_log.csv")
        # TEMPORARY until the input dataset is updated with the file training_log.csv.
        except FileNotFoundError:
            pass
    else:
        initial_epoch = 0

max_duration = MaxDuration()
        
epochs = full_dataset_passes * (len(sequences) // steps_per_epoch)    

print("Starting training.")
fit = model.fit(sequences, epochs = epochs, shuffle = shuffle_dataset, steps_per_epoch = steps_per_epoch,
    initial_epoch = initial_epoch,
    verbose = 2,
    use_multiprocessing = False,
    callbacks = [
        max_duration,
        LearningRate(),
        keras.callbacks.ModelCheckpoint("saved_model", save_freq = "epoch", save_weights_only = True, save_best_only = True),
        keras.callbacks.CSVLogger("training_log.csv", append = load_train_checkpoint)
        # keras.callbacks.ProgbarLogger("steps", verbose = 2)
    ])

print("Done training. Saving the model…")

model.save_weights("saved_model")

with open("last_epoch", "w") as f:
    f.write(str(max_duration.last_epoch))

print("Model saved.")

