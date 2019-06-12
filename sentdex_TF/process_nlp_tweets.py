import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd 

lemmatizer = WordNetLemmatizer()
encoding = 'latin-1'

def init_process(file, file_out):
    print('Initializing {}'.format(file))
    outfile = open(file_out, 'a')
    with open(file, buffering=200000, encoding=encoding) as f:
        try:
            for line in f:
                line = line.replace('"', '')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity  = [1, 0]
                elif initial_polarity == '4':
                    initial_polarity = [0, 1]
                
                tweet = line.split(',')[-1]
                outline = str(initial_polarity)+':::'+tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()
    print('{} created'.format(file_out))


def create_lexicon(file, len_w=2500):
    print('Creating lexicon...')
    lexicon = []
    with open(file, 'r', buffering=100000, encoding=encoding) as f:
        try: 
            counter = 1
            content = ''
            for line in f:
                counter += 1
                if(counter/float(len_w)).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' '+tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
        except Exception as e:
            print(str(e))
    
    print('Lexicon created with a length of {}'.format(len(lexicon)))
    file_name = 'lexicon.pickle'
    print('{} file created.'.format(file_name))
    with open(file_name, 'wb') as f:
        pickle.dump(lexicon, f)

    return file_name


def create_test_data_pickle(file):
    print('Creating test data')
    feature_sets = []
    labels  = []
    counter = 0
    with open(file, buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))

                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except Exception as e:
                print(e)
    
    feature_sets = np.array(feature_sets)
    labels  = np.array(labels)


def convert_to_vec(file, file_out, lexicon_pickle):
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)
    
    outfile = open(file_out, 'a')
    with open(file, buffering=20000, encoding=encoding) as f:
        counter = 0
        for line in f:
            counter += 1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            current_words  = word_tokenize(tweet.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words] 

            features = np.zeros(len(lexicon))

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            
            features = list(features)
            outline = str(features)+'::'+str(label)+'\n'
            outfile.write(outline)
        
    print(counter)


def shuffle_data(file):
    df = pd.read_csv(file, error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    df.to_csv('train_set_shuffled.csv', index=False)


if __name__ == '__main__':
    # init_process('training.1600000.processed.noemoticon.csv', 'train_set.csv')
    # init_process('testdata.manual.2009.06.14.csv', 'test_set.csv')
    # f_name = create_lexicon('train_set.csv', 2500)
    # shuffle_data('train_set.csv')
    convert_to_vec('test_set.csv','processed-test-set.csv','lexicon.pickle')
    create_test_data_pickle('processed-test-set.csv')