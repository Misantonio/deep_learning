import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time

lemmatizer = WordNetLemmatizer()
with open('lexicon.pickle','rb') as f:
    lexicon = pickle.load(f)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_classes = 2
batch_size = 100
total_batches = int(1600000/batch_size)
hm_epochs = 50
input_size = len(lexicon)
comp_time = 100.

x = tf.placeholder('float')
y = tf.placeholder('float')


def neural_network_model(data):

    hidden_1_layer = {
                    'weight':tf.Variable(tf.random_normal([input_size, n_nodes_hl1])),
                    'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {
                    'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                    'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {
                    'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'bias':tf.Variable(tf.random_normal([n_classes])),}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    saver = tf.train.Saver()
    tf_log = 'tf.log'

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:',epoch)
            start_epoch = epoch
        except:
            epoch = 1
            start_epoch = epoch

        batches_prev = time.time()
        start = time.time()
        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess,"model.ckpt")
            epoch_loss = 1

            if epoch == start_epoch:
                print('#### Computing remaining time... ####')

            with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            # OR DO +=1, test both
                            features[index_value] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                                  y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run +=1

                        if (batches_run/comp_time).is_integer():
                            batches_af = time.time()
                            batches_time = (batches_af-batches_prev)/3600.
                            rem_time = batches_time*(total_batches-batches_run)/comp_time
                            print('\nBatch run:',batches_run,'/',total_batches,'| Epoch:',epoch,'| Batch Loss:',c,)
                            print('{} batches took {} seconds'.format(comp_time, batches_time*3600))
                            print('Estimated remaining time: {} hours'.format(rem_time*(hm_epochs-epoch-1)))
                            batches_prev = batches_af

            saver.save(sess, "./model.ckpt")
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n') 
            epoch +=1
        end = time.time()
        print('\n \n Total training time: {} hours'.format((start-end)/3600.))


def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess,"model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0
        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
        print('Tested',counter,'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


if __name__ == '__main__':
    # train_neural_network(x)
    test_neural_network()