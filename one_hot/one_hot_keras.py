from numpy import array, argmax
from keras.utils import to_categorical

data = array([1,2,4,0,2,3,3])

encoded = to_categorical(data)
print(encoded)

inverted = argmax(encoded[0])
print(inverted)
