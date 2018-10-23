
import tensorflow as tf
from tensorflow.python.layers.core import Dense

class build_network:

    def __init__(self, params):
        self.rnn_size = params['rnn_size']
        self.num_layers = params['num_layers']
        self.encoding_embedding_size = params['encoding_embedding_size']
        self.decoding_embedding_size = params['decoding_embedding_size']
        self.learning_rate = params['learning_rate']
        self.keep_probability = params['keep_probability']
        #self.target_vocab_to_int = target_vocab_to_int

    # Input
    def model_inputs(self):
        """
        Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
        :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
        max target sequence length, source sequence length)
        """
        inputs = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name = 'targets')
        learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        tsl = tf.placeholder(tf.int32, [None], name='target_sequence_length')
        ssl = tf.placeholder(tf.int32, [None], name='source_sequence_length')
        mts = tf.reduce_max(tsl, name='max_target_len')
        return inputs, targets, learning_rate, keep_prob, tsl, mts, ssl

    # Decoder Input
    def process_decoder_input(self, target_data, target_vocab_to_int, batch_size):
        """
        Preprocess target data for encoding
        :param target_data: Target Placehoder
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :param batch_size: Batch Size
        :return: Preprocessed target data
        """
        go = target_vocab_to_int['<GO>']
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        decoded = tf.concat([tf.fill([batch_size, 1], go), ending], 1)

        return decoded

    # Encoding layer
    def encoding_layer(self, rnn_inputs, source_sequence_length, source_vocab_size):
        """
        Create encoding layer
        :param rnn_inputs: Inputs for the RNN
        :param source_sequence_length: a list of the lengths of each sequence in the batch
        :param source_vocab_size: vocabulary size of source data
        :return: tuple (RNN output, RNN state)
        """
        enc_embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, self.encoding_embedding_size)

        # RNN cell
        def make_cell(rnn_size):
            enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return enc_cell


        enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(self.rnn_size) for _ in range(self.num_layers)])
        enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=self.keep_probability)
        enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length,
                                                  dtype=tf.float32)

        return enc_output, enc_state

    # Deconding layer - training
    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input,
                             target_sequence_length, max_summary_length,
                             output_layer):
        """
        Create a decoding layer for training
        :param encoder_state: Encoder State
        :param dec_cell: Decoder RNN Cell
        :param dec_embed_input: Decoder embedded input
        :param target_sequence_length: The lengths of each sequence in the target batch
        :param max_summary_length: The length of the longest sequence in the batch
        :param output_layer: Function to apply the output layer
        :return: BasicDecoderOutput containing training logits and sample_id
        """
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                       sequence_length=target_sequence_length)
        basicDecoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                           helper=helper,
                                                           initial_state=encoder_state,
                                                           output_layer=output_layer)
        basicDecoderOutput = tf.contrib.seq2seq.dynamic_decode(basicDecoder,
                                                                   maximum_iterations=max_summary_length)[0]
        return basicDecoderOutput


    def decoding_layer_infer(self, encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                             end_of_sequence_id, max_target_sequence_length,
                             output_layer, batch_size):
        """
        Create a decoding layer for inference
        :param encoder_state: Encoder state
        :param dec_cell: Decoder RNN Cell
        :param dec_embeddings: Decoder embeddings
        :param start_of_sequence_id: GO ID
        :param end_of_sequence_id: EOS Id
        :param max_target_sequence_length: Maximum length of target sequences
        :param output_layer: Function to apply the output layer
        :param batch_size: Batch size
        :return: BasicDecoderOutput containing inference logits and sample_id
        """
        start_tokens = tf.constant([start_of_sequence_id] * batch_size)
        drop = tf.contrib.rnn.DropoutWrapper(dec_cell, self.keep_probability)
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens=start_tokens,
                                                              end_token=end_of_sequence_id)
        decoder = tf.contrib.seq2seq.BasicDecoder(drop, helper, encoder_state, output_layer)
        basicDecoderOutput = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_target_sequence_length)[0]
        return basicDecoderOutput


    def decoding_layer(self, dec_input, encoder_state,
                       target_sequence_length, max_target_sequence_length,
                       target_vocab_to_int, target_vocab_size, batch_size):
        """
        Create decoding layer
        :param dec_input: Decoder input
        :param encoder_state: Encoder state
        :param target_sequence_length: The lengths of each sequence in the target batch
        :param max_target_sequence_length: Maximum length of target sequences
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :param target_vocab_size: Size of target vocabulary
        :param batch_size: The size of the batch
        :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
        """
        dec_embeddings = tf.Variable(tf.random_normal([target_vocab_size, self.decoding_embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        def lstm_cell():
            lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            return tf.contrib.rnn.DropoutWrapper(lstm, self.keep_probability)

        dec_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.num_layers)])

        output_layer = Dense(target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        with tf.variable_scope('decode'):
            train_logits = self.decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length,
                                                max_target_sequence_length, output_layer)

        with tf.variable_scope('decode', reuse=True):
            infer_logits = self.decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, target_vocab_to_int['<GO>'],
                                                target_vocab_to_int['<EOS>'], max_target_sequence_length, output_layer, batch_size)

        return train_logits, infer_logits


    def seq2seq_model(self, input_data, target_data, batch_size,
                      source_sequence_length, target_sequence_length,
                      max_target_sentence_length,
                      source_vocab_size, target_vocab_size, target_vocab_to_int):
        """
        Build the Sequence-to-Sequence part of the neural network
        :param input_data: Input placeholder
        :param target_data: Target placeholder
        :param keep_prob: Dropout keep probability placeholder
        :param batch_size: Batch Size
        :param source_sequence_length: Sequence Lengths of source sequences in the batch
        :param target_sequence_length: Sequence Lengths of target sequences in the batch
        :param source_vocab_size: Source vocabulary size
        :param target_vocab_size: Target vocabulary size
        :param enc_embedding_size: Decoder embedding size
        :param dec_embedding_size: Encoder embedding size
        :param rnn_size: RNN Size
        :param num_layers: Number of layers
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
        """
        _, enc_state = self.encoding_layer(input_data,
                                      source_sequence_length,
                                      source_vocab_size)
        dec_input = self.process_decoder_input(target_data,
                                          target_vocab_to_int,
                                          batch_size)
        training_output, inference_output = self.decoding_layer(dec_input,
                                                           enc_state,
                                                           target_sequence_length,
                                                           max_target_sentence_length,
                                                           target_vocab_to_int,
                                                           target_vocab_size,
                                                           batch_size)
        return (training_output, inference_output)

