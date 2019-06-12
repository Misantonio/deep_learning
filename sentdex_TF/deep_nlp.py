import tensorflow as tf
from process_nlp import create_features_sets_and_labels
import numpy as np

train_x, train_y, test_x, test_y = create_features_sets_and_labels('pos.txt', 'neg.txt')

input_size = len(train_x[0])
nodes_hl1 = 500
nodes_hl2 = 500
nodes_hl3 = 500
n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float')

def network_model(data):
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([input_size, nodes_hl1])), 
                      'biases': tf.Variable(tf.random_normal([nodes_hl1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])), 
                      'biases': tf.Variable(tf.random_normal([nodes_hl2]))}
    
    # hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])), 
    #                   'biases': tf.Variable(tf.random_normal([nodes_hl3]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl2, n_classes])), 
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.matmul(data, hidden_layer_1['weights']) + hidden_layer_1['biases']
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1, hidden_layer_2['weights']) + hidden_layer_2['biases']
    l2 = tf.nn.relu(l2)

    # l3 = tf.matmul(l2, hidden_layer_3['weights']) + hidden_layer_3['biases']
    # l3 = tf.nn.relu(l3)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

    return output

def train_network(data):
    prediction = network_model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()

    hm_epochs = 15
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):           
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start: end])
                batch_y = np.array(train_y[start: end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
        if input('Would like to save? [y/n]') == 'y':
            saver.save(sess, './model.ckpt')
            print('Model saved.')

train_network(x)
