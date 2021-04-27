import tensorflow.keras as keras
import saint_plus

# Warmup steps
warmup_steps = 4000


###################
# Train the model #
###################

class LearningRate(keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        if batch > 0:
            lr = saint_plus.d_model ** -0.5 * min(batch ** -0.5, batch * warmup_steps ** -1.5)
        else:
            lr = 0
        keras.backend.set_value(self.model.optimizer.lr, lr)

def track_time_to_file(desc, x):
    with open("timetrack_train.csv", "a") as f:
        f.write("{}, {}\n".format(time.time(), x))
        f.flush()

track_time = keras.callbacks.LambdaCallback(
    on_batch_begin = lambda batch, logs: track_time_to_file("Batch", "+1"),
    on_batch_end = lambda batch, logs: track_time_to_file("Batch", "-1")
)

print("Starting training.")
fit = saint_plus.model.fit(saint_plus.sequences, epochs = 5, shuffle = True, use_multiprocessing = False,
    callbacks = [
        # track_time,
        LearningRate(),
        keras.callbacks.ModelCheckpoint("saved_model", save_freq = 500, save_weights_only = True),
        keras.callbacks.ProgbarLogger("steps")
    ])



