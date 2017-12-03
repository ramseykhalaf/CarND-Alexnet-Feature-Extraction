import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.

training_file = 'train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X = train['features']
y = train['labels']
nb_classes = 43

# TODO: Split data into training and validation sets.

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# TODO: Define placeholders and resize operation.



# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(..., feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.

shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

print(shape)
weights = tf.Variable(tf.truncated_normal(shape=shape, seed=123, dtype=tf.float32))
biases = tf.Variable(tf.zeros(nb_classes), dtype=tf.float32)

logits = tf.matmul(fc7, weights) + biases
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: Train and evaluate the feature extraction model.
