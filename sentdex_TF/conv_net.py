import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

input_size = 784
nodes_hl1 = 200
nodes_hl2 = 200
nodes_hl3 = 200
n_classes = 10
hm_epochs  = 10

batch_size = 100

x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float', [None, n_classes])

def conv2d(data, W):
    return tf.nn.conv2d(data, W, strides=[1,1,1,1], padding='SAME')


def maxpool2d(data):
    return tf.nn.max_pool(data, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def network_model(data):
    weights = {
        # 5x5 convolution, 1 input image, 32 outputs
        'conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        'conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        'fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'conv1': tf.Variable(tf.random_normal([32])),
        'conv2': tf.Variable(tf.random_normal([64])),
        'fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Reshape input to a 4D tensor
    data = tf.reshape(data, shape=[-1, 28, 28, 1])
    # Convolution layer
    conv1 = tf.nn.relu(conv2d(data, weights['conv1'])+biases['conv1'])
    conv1 = maxpool2d(conv1)
    # Convolution layer
    conv2 = tf.nn.relu(conv2d(conv1, weights['conv2'])+biases['conv2'])
    conv2 = maxpool2d(conv2)
    # Fully connected layer
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['fc']+biases['fc']))
    # Output layer
    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


def train_network(data):
    prediction = network_model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        batch_tx, batch_ty = mnist.test.next_batch(2000)
        print('Accuracy:', accuracy.eval(feed_dict={x: batch_tx, y: batch_ty}))

if __name__ == '__main__':
    train_network(x)
