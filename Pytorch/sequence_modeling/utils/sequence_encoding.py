import numpy as np
from sklearn.preprocessing import OneHotEncoder


def onehot_sequence_encoder(df, column, max_seq_len):
    """One hot encoder of amino acid sequence data.

    When a sequence is padded, (+) value is assumed.

    :param df: dataframe containing sequence with sequence_a_padded column.
    :param column: column of amino acid padded sequence.
    :param max_seq_len: max length of amino acid sequence with padding included.
    :return: array of one hot encoded sequences.
    """
    aa_symbols = list('ACDEFGHIKLMNPQRSTVWY+')
    aa_enc_template = np.array([[aa] * max_seq_len for aa in aa_symbols])
    aa_enc = OneHotEncoder().fit(aa_enc_template)

    seq_array = np.array([list(seq) for seq in df[column]])
    seq_encoded = aa_enc.transform(seq_array).toarray()
    return seq_encoded


def atchley_sequence_encoder():
    pass


def blosum65_sequence_encoder():
    pass