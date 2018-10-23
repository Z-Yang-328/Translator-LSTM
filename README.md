# Translator-LSTM

This repository apply LSTM, one derivative of RNN, which was introduced to public in 1997 by Dr. Sepp Hochreiter and Dr. Jürgen Schmidhuber, to build a English-French translator. Since training on the whole language of English and French will take days to complete, this program use a small portion of English corpus to train the model.

### RNN
Recurrent Neural Network, or RNN, is one of the most useful neural network structure in the world. This sequence-structure are used to build models in speech recognition and language translation. As you know, the two tasks I mentioned are all about sequence data and need a model to figure out their sequence structures.

### LSTM
Long-Short-Term Memory, known as LSTM, is one derivative of original RNN. LSTM can, as its name shows, store infomation learned from the data like its "memory". Just like human memory, the memory of LSTM enables the model to learn from new data based on their memory, instead of training the model from zero. This mechanism dramtically increase the models' performance and reduce the training time.

### Try with more data
If you would like to train the model on the whole language, play with [WMT10 French-English corpus!]http://www.statmt.org/wmt10/training-giga-fren.tar to train your model. One thing you should notice before you start: this will take DAYS to train your model even on GPU.

### Files descriptions
* assistance.py: Functions to load and process data.
* preprocessing.py: Function to preprocess data, the obtained data can feed in the model.
* check_tensorflow_gpu.py: To check Tensorflow version and access to GPU.
* build_network.py: Steps to build the network and build sequence models. Define each layers in the network.
* train_network.py: To train the network using functions built in build_network.py.
* main.py: To put everything together, download all other files and run main.py to build the translator.
* translator.ipynb: An easy-to-present version of the project. Check out the data and results in the notebook.
