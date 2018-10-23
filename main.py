
import assistance

from preprocessing import text_to_ids

params = {
            'epochs': 10,
            'batch_size': 256,
            'rnn_size': 256,
            'num_layers': 3,
            'encoding_embedding_size': 128,
            'decoding_embedding_size': 128,
            'learning_rate': 0.001,
            'keep_probability': 0.5,
            'display_step': 50
         }
source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'

assistance.preprocess_and_save_data(source_path, target_path, text_to_ids)

from train_network import training

tn = training(params)
tn.build_graph()
tn.train_network()