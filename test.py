# from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.clear_session()  # For easy reset of notebook state.


inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
#
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=4)
# Reset metrics before saving so that loaded model has same state,
# since metric states are not preserved by Model.save_weights
model.reset_metrics()
predictions = model.predict(x_test)

"""Whole-model saving
You can save a model built with the Functional API into a single file.
You can later recreate the same model from this file, even if you
 no longer have access to the code that created the model.

This file includes:

-> The model's architecture
-> The model's weight values (which were learned during training)
-> The model's training config (what you passed to compile), if any
-> The optimizer and its state, if any (this enables you to restart training where you left)"""

# Save the model
model.save('path_to_my_model.h5')

# Recreate the exact same model purely from the file
new_model = keras.models.load_model('path_to_my_model.h5')

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
print(np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6))

# Note that the optimizer state is preserved as well:
# you can resume training where you left off.
new_model.fit(x_train, y_train,
              batch_size=64,
              epochs=4)
new_predictions = new_model.predict(x_test)
# Export the model to a SavedModel
new_model.save('path_to_saved_model', save_format='tf')

# Recreate the exact same model
new_model_2 = keras.models.load_model('path_to_saved_model')

# Check that the state is preserved
new_predictions_2 = new_model_2.predict(x_test)
np.testing.assert_allclose(new_predictions, new_predictions_2, rtol=1e-6, atol=1e-6)

new_model_2.fit(x_train, y_train,
                batch_size=64,
                epochs=4)

# Note that the optimizer state is preserved as well:
# you can resume training where you left off.

"""Architecture-only saving
Sometimes, you are only interested in the architecture of the model,
and you don't need to save the weight values or the optimizer. In this case,
you can retrieve the "config" of the model via the get_config() method.
The config is a Python dict that enables you to recreate the same model -- initialized from scratch,
without any of the information learned previously during training.
-> The model's architecture
 X The model's weight values (which were learned during training)
 X The model's training config (what you passed to compile), if any
 X The optimizer and its state, if any (this enables you to restart training where you left)"""


config = model.get_config()
reinitialized_model = keras.Model.from_config(config)

# Note that the model state is not preserved! We only saved the architecture.
new_predictions = reinitialized_model.predict(x_test)
assert abs(np.sum(predictions - new_predictions)) > 0.

"""Weights-only saving
Sometimes, you are only interested in the state of the model -- its weights values --
and not in the architecture. In this case, you can retrieve the weights values as a list
 of Numpy arrays via get_weights(), and set the state of the model via set_weights:
 X The model's architecture
-> The model's weight values (which were learned during training)
 X The model's training config (what you passed to compile), if any
 X The optimizer and its state, if any (this enables you to restart training where you left)"""

weights = model.get_weights()  # Retrieves the state of the model.
model.set_weights(weights)  # Sets the state of the model.

""" get_config()/from_config() and get_weights()/set_weights()
You can combine get_config()/from_config() and get_weights()/set_weights() to recreate your model in the same state.
However, UNLIKE model.save(), this will NOT INCLUDE the training config and the optimizer.
You would have to call compile() again before using the model for training.
-> The model's architecture
-> The model's weight values (which were learned during training)
 X The model's training config (what you passed to compile), if any
 X The optimizer and its state, if any (this enables you to restart training where you left)"""


config = model.get_config()
weights = model.get_weights()

new_model = keras.Model.from_config(config)
new_model.set_weights(weights)

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# Note that the optimizer was not preserved,
# so the model should be compiled anew before training
# (and the optimizer will start from a blank state).


""" model.to_json() model_from_json() + save_weights(fpath) and load_weights(fpath)
The save-to-disk alternative to get_weights() and set_weights(weights) is
 save_weights(fpath) and load_weights(fpath). Here's an example that saves to disk:
-> The model's architecture
-> The model's weight values (which were learned during training)
 X The model's training config (what you passed to compile), if any
 X The optimizer and its state, if any (this enables you to restart training where you left)"""

# Save JSON config to disk
json_config = model.to_json()
with open('model_config.json', 'w') as json_file:
    json_file.write(json_config)
# Save weights to disk
model.save_weights('path_to_my_weights.h5')

# Reload the model from the 2 files we saved
with open('model_config.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights('path_to_my_weights.h5')

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# Note that the optimizer was not preserved.

"""But remember that the simplest, recommended way is just this:"""

model.save('path_to_my_model.h5')
del model
model = keras.models.load_model('path_to_my_model.h5')
