#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from music21 import *


# In[2]:


# load the datasets and then split in to train/test data sets
# this all has to be done at the same time to ensure that the durations/pitches are from the same samples
with open("pickles/duration_X_train.pickle", 'rb') as duration_x, open('pickles/duration_Y_train.pickle', 'rb') as duration_y,    open("pickles/pitch_X_train.pickle", 'rb') as pitch_x, open('pickles/pitch_Y_train.pickle', 'rb') as pitch_y:
    duration_X_train, duration_X_test, duration_Y_train, duration_Y_test,    pitch_X_train, pitch_X_test, pitch_Y_train, pitch_Y_test    = train_test_split(pickle.load(duration_x), pickle.load(duration_y),                       pickle.load(pitch_x), pickle.load(pitch_y), train_size=0.95)


# In[3]:


# retrieve the pitches and durations that were used to build the data set
# these will be used to convert the output one-hot vectors back to actual pitch/duration values
with open('pickles/durations.pickle', 'rb') as d, open('pickles/pitches.pickle', 'rb') as p:
    durations = pickle.load(d)
    pitches = pickle.load(p)


# In[4]:


num_samples = duration_X_train.shape[0]  # applies to both pitch and duration data sets
timesteps = duration_X_train.shape[1]  # applies to both pitch and duration data sets
duration_num_features = duration_X_train.shape[2]
pitch_num_features = pitch_X_train.shape[2]


# In[5]:


# single layer Unidirectional or Bidirectional LSTM; will easily allow us to test various configurations
def getModel(t, num_features, bidirectional=True):
    model = Sequential()
    # only dif. betwn. bi. LSTM and uni. LSTM is the presence/absence of Bidirectional wrapper
    # hidden layer 1; 20  units; input (# timesteps, # features); return a sequence of each time step's outputs
    if bidirectional:
        model.add(Bidirectional(LSTM(20, input_shape=(t, num_features), return_sequences=True)))
    else:
        model.add(LSTM(20, input_shape=(t, num_features), return_sequences=True))
        
    # TimeDistributed is a wrapper allowing one output per time step; 
    # ...requires hidden layer to have return_sequences == True
    model.add(TimeDistributed(Dense(num_features, activation='softmax')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])
    return model


# In[ ]:


# train LSTM
def trainModel(model, X, Y, epochs=10):
    for epoch in range(epochs):
        # generate new random sequence
        # fit model for one epoch on this sequence
        model.fit(X, Y, epochs=epochs, verbose=1, validation_split=0.2)


# In[ ]:


duration_bidirectional = getModel(timesteps, duration_num_features, bidirectional=True)
trainModel(duration_bidirectional, duration_X_train, duration_Y_train)


# In[ ]:


pitch_bidirectional = getModel(timesteps, pitch_num_features, bidirectional=True)
trainModel(pitch_bidirectional, pitch_X_train, pitch_Y_train)


# In[ ]:


# model.predict requires 3D vector, so this reshapes a 2D input to 3D so it can be fed through the network
def to_3D(sample):
    return sample.reshape(1, sample.shape[0], sample.shape[1])


# In[ ]:


# converts output one-hot vectors to its respective quarter length value (based on durations seen in data set)
def duration_one_hot_to_quarter_length(prediction):
    new_durations = []
    for timestep in prediction:
        index = np.argmax(timestep)
        new_durations.append(durations[index])
    
    return new_durations

# converts output one-hot vectors to its respective MIDI pitch value (based on durations seen in data set)
def pitch_one_hot_to_MIDI(prediction):
    new_pitches = []
    for timestep in prediction:
        index = np.argmax(timestep)
        new_pitches.append(pitches[index])
    
    return new_pitches


# In[ ]:


duration_pred = duration_bidirectional.predict(to_3D(duration_X_test[0])).reshape(timesteps, duration_num_features)
composed_durations = duration_one_hot_to_quarter_length(duration_pred)

pitch_pred = pitch_bidirectional.predict(to_3D(pitch_X_test[0])).reshape(timesteps, pitch_num_features)
composed_pitches = pitch_one_hot_to_MIDI(pitch_pred)


# In[ ]:


composed_stream = stream.Stream()
for pair in composed_pairs:
    p = pitch.Pitch(midi=pair[0])
    d = duration.Duration(pair[1])
    n = note.Note()
    n.pitch = p
    n.duration = d
    composed_stream.append(n)


# In[ ]:


composed_stream.show('midi')


# In[ ]:




