
import tensorflow as tf
import assistance
import numpy as np

from build_network import build_network

# Build Graph
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = assistance.load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

class training:

    def __init__(self, params):
        self.params = params
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.display_step = params['display_step']
        self.train_graph = None
        self.train_op = None
        self.train_source = None
        self.train_target = None
        self.valid_sources_batch = None
        self.valid_targets_batch = None
        self.valid_sources_lengths = None
        self.valid_targets_lengths = None
        self.cost = None
        self.input_data = None
        self.targets = None
        self.keep_prob = None
        self.target_sequence_length = None
        self.max_target_sequence_length = None
        self.source_sequence_length = None
        self.lr = None
        self.train_logits = None
        self.inference_logits = None
        self.bn = build_network(params)

    def build_graph(self):
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            self.input_data, self.targets, self.lr, self.keep_prob, self.target_sequence_length, self.max_target_sequence_length, self.source_sequence_length = self.bn.model_inputs()

            #sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
            input_shape = tf.shape(self.input_data)

            train_logits, self.inference_logits = self.bn.seq2seq_model(tf.reverse(self.input_data, [-1]),
                                                           self.targets,
                                                           self.batch_size,
                                                           self.source_sequence_length,
                                                           self.target_sequence_length,
                                                           self.max_target_sequence_length,
                                                           len(source_vocab_to_int),
                                                           len(target_vocab_to_int),
                                                           target_vocab_to_int)


            training_logits = tf.identity(train_logits.rnn_output, name='logits')
            self.inference_logits = tf.identity(self.inference_logits.sample_id, name='predictions')

            masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')

            with tf.name_scope("optimization"):
                # Loss function
                self.cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    self.targets,
                    masks)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(self.lr)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(self.cost)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gradients)


    def pad_sentence_batch(self, sentence_batch, pad_int):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


    def get_batches(self, sources, targets, source_pad_int, target_pad_int):
        """Batch targets, sources, and the lengths of their sentences together"""
        for batch_i in range(0, len(sources)//self.batch_size):
            start_i = batch_i * self.batch_size

            # Slice the right amount for the batch
            sources_batch = sources[start_i:start_i + self.batch_size]
            targets_batch = targets[start_i:start_i + self.batch_size]

            # Pad
            pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch, source_pad_int))
            pad_targets_batch = np.array(self.pad_sentence_batch(targets_batch, target_pad_int))

            # Need the lengths for the _lengths parameters
            pad_targets_lengths = []
            for target in pad_targets_batch:
                pad_targets_lengths.append(len(target))

            pad_source_lengths = []
            for source in pad_sources_batch:
                pad_source_lengths.append(len(source))

            yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

    def get_accuracy(self, target, logits):
        """
        Calculate accuracy
        """
        max_seq = max(target.shape[1], logits.shape[1])
        if max_seq - target.shape[1]:
            target = np.pad(
                target,
                [(0,0),(0,max_seq - target.shape[1])],
                'constant')
        if max_seq - logits.shape[1]:
            logits = np.pad(
                logits,
                [(0,0),(0,max_seq - logits.shape[1])],
                'constant')

        return np.mean(np.equal(target, logits))

    # Split data to training and validation sets
    def get_train_test(self):
        self.train_source = source_int_text[self.batch_size:]
        self.train_target = target_int_text[self.batch_size:]
        valid_source = source_int_text[:self.batch_size]
        valid_target = target_int_text[:self.batch_size]
        (self.valid_sources_batch, self.valid_targets_batch, self.valid_sources_lengths, self.valid_targets_lengths ) = next(self.get_batches(valid_source,
                                                                                                                     valid_target,
                                                                                                                     source_vocab_to_int['<PAD>'],
                                                                                                                     target_vocab_to_int['<PAD>']))


    def train_network(self):
        with tf.Session(graph=self.train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.get_train_test()
            for epoch_i in range(self.epochs):
                for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                        self.get_batches(self.train_source, self.train_target,
                                    source_vocab_to_int['<PAD>'],
                                    target_vocab_to_int['<PAD>'])):

                    _, loss = sess.run(
                        [self.train_op, self.cost],
                        {self.input_data: source_batch,
                         self.targets: target_batch,
                         self.lr: self.params['learning_rate'],
                         self.target_sequence_length: targets_lengths,
                         self.source_sequence_length: sources_lengths,
                         self.keep_prob: self.params['keep_probability']})


                    if batch_i % self.display_step == 0 and batch_i > 0:


                        batch_train_logits = sess.run(
                            self.inference_logits,
                            {self.input_data: source_batch,
                             self.source_sequence_length: sources_lengths,
                             self.target_sequence_length: targets_lengths,
                             self.keep_prob: 1.0})


                        batch_valid_logits = sess.run(
                            self.inference_logits,
                            {self.input_data: self.valid_sources_batch,
                             self.source_sequence_length: self.valid_sources_lengths,
                             self.target_sequence_length: self.valid_targets_lengths,
                             self.keep_prob: 1.0})

                        train_acc = self.get_accuracy(target_batch, batch_train_logits)

                        valid_acc = self.get_accuracy(self.valid_targets_batch, batch_valid_logits)

                        print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                              .format(epoch_i+1, batch_i, len(source_int_text) // self.batch_size, train_acc, valid_acc, loss))

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print('Model Trained and Saved')