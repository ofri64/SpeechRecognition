import numpy as np

EMPTY = "-"  # choose some token that isn't likely to represent a phoneme


def load_matrix(matrix_file):
    return np.load(matrix_file)


def get_modified_transcript(original_transcript):
    return "".join([EMPTY + t for t in original_transcript] + [EMPTY])


def get_alphabet_index_translation(alphabet):
    alphabet_empty_token = alphabet + EMPTY  # add the empty token
    alphabet_indices = enumerate(alphabet_empty_token)
    alphabet_translation = {token: index for index, token in alphabet_indices}  # token -> index translation

    return alphabet_translation
