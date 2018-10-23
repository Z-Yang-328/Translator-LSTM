

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    eos = source_vocab_to_int['<EOS>']
    source_id_text = [[source_vocab_to_int[word] for word in sentence.split()] for sentence in source_text.split('\n')]
    target_id_text = [[target_vocab_to_int[word] for word in sentence.split()] + [eos] for sentence in target_text.split('\n')]

    return source_id_text, target_id_text