import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

input_size = 784
nodes_hl1 = 200
nodes_hl2 = 200
nodes_hl3 = 200
n_classes = 10

batch_size = 100

x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float')

def network_model(data):
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([input_size, nodes_hl1])), 
                      'biases': tf.Variable(tf.random_normal([nodes_hl1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])), 
                      'biases': tf.Variable(tf.random_normal([nodes_hl2]))}
    
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])), 
                      'biases': tf.Variable(tf.random_normal([nodes_hl3]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl3, n_classes])), 
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.matmul(data, hidden_layer_1['weights']) + hidden_layer_1['biases']
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1, hidden_layer_2['weights']) + hidden_layer_2['biases']
    l2 = tf.nn.relu(l2)

    l3 = tf.matmul(l2, hidden_layer_3['weights']) + hidden_layer_3['biases']
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_network(data):
    prediction = network_model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 20

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
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_network(x)
