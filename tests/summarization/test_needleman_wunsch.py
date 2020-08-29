import json
import os
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from crd3.summarization.needleman_wunsch import NeedlemanWunschAligner, TextChunk
from crd3.utils.temp_utils import calculate_scaled_f1_turn_distribution, nltk_stopword_tokenize

FILE_PATH = Path(__file__).absolute()


def test_build_scoring_matrix():
    sample_summary_path = os.path.join(os.path.dirname(FILE_PATH), 'sample_summary.json')
    sample_turns_path = os.path.join(os.path.dirname(FILE_PATH), 'sample_turns.json')
    sample_score_matrix_path = os.path.join(os.path.dirname(FILE_PATH), 'sample_score_matrix.csv')

    chunks = [TextChunk(id=elem['ID'], text=elem['CHUNK']) for elem in
              json.loads(open(sample_summary_path, encoding='utf-8').read())]
    turns = [TextChunk(id=elem['ID'], text=elem['TURN']) for elem in
             json.loads(open(sample_turns_path, encoding='utf-8').read())]
    tokenized_chunks = [set(nltk_stopword_tokenize(chunk.text, 2, skip_unigrams=False)) for chunk in chunks]
    tokenized_turns = [set(nltk_stopword_tokenize(turn.text, 2, skip_unigrams=False)) for turn in turns]

    s = NeedlemanWunschAligner.build_scoring_matrix(tokenized_turns, tokenized_chunks,
                                                    calculate_scaled_f1_turn_distribution)
    sample = np.genfromtxt(sample_score_matrix_path, delimiter=",")
    assert_array_equal(s, sample)

def test_needleman_wunsch_algorithm():
    sample_score_matrix_path = os.path.join(os.path.dirname(FILE_PATH), 'sample_score_matrix.csv')
    sample_nw_output_path = os.path.join(os.path.dirname(FILE_PATH), 'sample_nw_algorithm_output.json')

    matrix = np.genfromtxt(sample_score_matrix_path, delimiter=",")
    summary_sequence = list(range(matrix.shape[1]))
    turn_sequence = list(range(matrix.shape[0]))

    nw_result = NeedlemanWunschAligner.needleman_wunsch_algorithm(turn_sequence, summary_sequence, matrix)

    expected_sequence_1 = []
    expected_sequence_2 = []
    expected_scores = []

    for expected_alignment in json.load(open(sample_nw_output_path)):
        expected_sequence_1.append(expected_alignment['sequence_1'])
        expected_sequence_2.append(expected_alignment['sequence_2'])
        expected_scores.append(expected_alignment['score'])

    assert nw_result.sequence_1 == expected_sequence_1
    assert nw_result.sequence_2 == expected_sequence_2
    assert nw_result.scores == expected_scores
