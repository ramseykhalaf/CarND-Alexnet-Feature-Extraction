import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from alexnet import AlexNet

# TODO: Load traffic signs data.
EPOCHS = 10

nb_classes = 43
batch_size = 64

training_file = 'train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X = train['features']
y = train['labels']

# TODO: Split data into training and validation sets.

X_train, X_val, y_train, y_val = train_test_split(X, y)

X_train = X_train[:1024]
y_train = y_train[:1024]

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

# TODO: Define placeholders and resize operation.

features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.

fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.

shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

print(shape)

weights = tf.Variable(tf.truncated_normal(shape))
biases = tf.Variable(tf.zeros(nb_classes), dtype=tf.float32)

logits = tf.matmul(fc7, weights) + biases
probs = tf.nn.softmax(logits)
preds = tf.arg_max(probs, dimension=1)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_op = tf.reduce_mean(cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss_op, var_list=[weights, biases])
init_op = tf.global_variables_initializer()

accuracy_op = tf.reduce_mean(tf.cast(tf.equal(labels, preds), tf.float32))

# TODO: Train and evaluate the feature extraction model.

def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: X_batch, labels: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])
    return total_loss/X.shape[0], total_acc/X.shape[0]


with tf.Session() as session:
    session.run(init_op)

    X_train, y_train = shuffle(X_train, y_train)
    for e in range(EPOCHS):
        print("Starting epoch: ", e)
        for offset in range(0, X_train.shape[0], batch_size):
            print("Batch with offset: ", offset)
            end = offset + batch_size
            train_batch_dict = {
                features: X_train[offset:end],
                labels: y_train[offset:end],
            }
            session.run(train_op, feed_dict=train_batch_dict)

        loss, acc = eval_on_data(X_val, y_val, session)

        print("epoch: ", e)
        print("loss: ", loss, " val accuracy: ", acc)



