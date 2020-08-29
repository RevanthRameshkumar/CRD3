from dataclasses import dataclass
from typing import List, Callable, TypeVar
import numpy as np
from multiprocessing import Pool
import os
from functools import partial
from crd3.utils.temp_utils import nltk_stopword_tokenize, calculate_scaled_f1_turn_distribution


T = TypeVar('T')


@dataclass
class TextChunk:
    id: str
    text: str


@dataclass
class NeedlemanWunschResult:
    sequence_1: List[T]
    sequence_2: List[T]
    scores: List[float]

    GAP_SYMBOL: str = '-'


def get_scores_for_entire_sequence(obj_1: T,
                                   obj_sequence: List[T],
                                   scoring_function: Callable[[T, T], float]) -> List[float]:
    return [scoring_function(obj_1, obj_2) for obj_2 in obj_sequence]


class NeedlemanWunschAligner:
    def __init__(self):
        pass

    @staticmethod
    def build_scoring_matrix(sequence_1: List[T],
                             sequence_2: List[T],
                             scoring_function: Callable[[T, T], float]) -> np.ndarray:
        """
        Build a score matrix using the two sequences as input, with each element in the matrix being filled by the
        scoring function specified by scoring_function.
        The top left of the matrix is the function of the 0th element of both sequences with sequence 1 on the y axis
        and sequence 2 on the x axis.
        Internally, it uses simple multi-threading to speed up the computation for long
        sequences (maps to as many available cores as possible).

        :return: returns a 2D scoring matrix using the sequence 1 as the y axis and sequence 2 as the x axis.
                 The top left is 0,0.
        :param sequence_1: the first list of text_chunks to score against
        :param sequence_2: the second list of text_chunks to score against
        :param scoring_function: any generic function f such that f(object_1, object_2) = float. The reason why we make
            it a function of generic objects is to give the option of tokenizing outside of the scoring
            function for better performance.
        """
        p = Pool(os.cpu_count() - 1)
        scored_pairs = p.map(partial(get_scores_for_entire_sequence,
                                     obj_sequence=sequence_1,
                                     scoring_function=scoring_function), sequence_2)
        score_matrix = np.transpose(np.array(scored_pairs))
        return score_matrix

    @staticmethod
    def needleman_wunsch_algorithm(sequence_1: List[str],
                                   sequence_2: List[str],
                                   scoring_matrix: np.ndarray) -> NeedlemanWunschResult:
        """
        Runs the Needleman Wunsch algorithm along the scoring matrix, with the y axis of the scoring matrix
        corresponding to sequence_1 and the x axis of the scoring matrix corresponding to sequence_2.

        :return: Returns a NeedlemanWunschResult where the sequences are the same order as the input sequences,
                 and they are aligned using '-' for blank alignment placeholder. The scores are cumulative scores from
                 the backtracking algorithm.
        :param sequence_1: the first sequence to align, corresponding to the y axis of the scoring matrix
        :param sequence_2: the second sequence to align, corresponding to the x axis of the scoring matrix
        :param scoring_matrix: a 2D scoring matrix of floats (any shape)
        """
        m = len(sequence_1)
        n = len(sequence_2)

        assert scoring_matrix.shape[0] == m  # make sure the length of sequence 1 is the same as the # of rows
        assert scoring_matrix.shape[1] == n  # and sequence 2 is same as # of columns

        result = NeedlemanWunschResult(sequence_1=[], sequence_2=[], scores=[])

        # initialize the alignment score matrix H, insert M and build the first row and column
        score_matrix = np.zeros((m + 1, n + 1))
        score_matrix[0, :] = np.arange(0, -1 * n - 1, -1)
        score_matrix[:, 0] = np.arange(0, -1 * m - 1, -1)
        score_matrix[1:, 1:] = scoring_matrix

        # the gap penalties are 0, but in the future this can be a parameter to change
        left_gap_penalty = 0
        down_gap_penalty = 0
        # fill out H with the cumulative scores, this is the forward filling step of the NW algorithm
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                current_score = score_matrix[i][j]
                match_score = score_matrix[i - 1][j - 1] + current_score
                delete_score = score_matrix[i - 1][j] + down_gap_penalty + current_score
                insert_score = score_matrix[i][j - 1] + left_gap_penalty + current_score
                score_matrix[i][j] = max(match_score, delete_score, insert_score)

        # using the cumulative score filled matrix, traceback to see which optimal path to take to align the sequences
        # this method only gives 1 path, but multiple paths are possible if they yield the same score, this extension
        #   can be added on later
        i = m
        j = n
        while i > 0 and j > 0:
            diagonal_score = score_matrix[i - 1, j - 1]
            up_score = score_matrix[i - 1, j]
            left_score = score_matrix[i, j - 1]

            direction = np.argmax([diagonal_score, up_score, left_score])
            if direction == 0:
                result.sequence_1.insert(0, sequence_1[i-1])
                result.sequence_2.insert(0, sequence_2[j-1])
                result.scores.insert(0, diagonal_score)
                i -= 1
                j -= 1
            elif direction == 1:
                result.sequence_1.insert(0, sequence_1[i - 1])
                result.sequence_2.insert(0, result.GAP_SYMBOL)
                result.scores.insert(0, up_score)
                i -= 1
            elif direction == 2:
                result.sequence_1.insert(0, result.GAP_SYMBOL)
                result.sequence_2.insert(0, sequence_2[j - 1])
                result.scores.insert(0, left_score)
                j -= 1

        while j > 0:
            j -= 1
            result.scores.insert(0, 0)
        while i > 0:
            i -= 1
            result.scores.insert(0, 0)
        return result

    def _process_alignments(self,
                            needleman_wunsch_result: NeedlemanWunschResult,
                            sequence_1: List[TextChunk],
                            sequence_2: List[TextChunk]) -> NeedlemanWunschResult:
        """
        Processes the input NeedlemanWunschResult and aligns the two input sequences, with '-' in the
        NeedlemanWunschResult resolving into a repeat. The scores are transformed from cumulative to corresponding to
        each (sequence_1[i], sequence_2[i]) pair.

        :return: a NeedlemanWunschResult where the sequences are of type TextChunk and the scores are that of the
                 individually aligned elements of the two sequences, not cumulative.
        :param needleman_wunsch_result: where the sequences are of type str and the blank alignment placeholder is '-'
        :param sequence_1: a list of TextChunks to align using the result from needleman_wunsch_result.sequence_1
        :param sequence_2: a list of TextChunks to align using the result from needleman_wunsch_result.sequence_2
        """
        pass

    def align_text_sequences(self,
                             sequence_1: List[TextChunk],
                             sequence_2: List[TextChunk],
                             scoring_method: str) -> NeedlemanWunschResult:
        """
        Takes 2 lists of text_chunks and aligns them using the Needleman-Wunsch alignment algorithm and the specified
        scoring method.

        :return: returns a NeedlemanWunschResult with sequences of type TextChunk aligned with each other.
                 The NeedlemanWunschResult sequence order is the same as the input sequence order.
        :param sequence_1: the first list of text_chunks to align
        :param sequence_2: the second list of text_chunks to align
        :param scoring_method: the method to use for scoring the sequences while building the scoring matrix.
        One of ['scaled_rouge_f1',]
        """
        if scoring_method == 'scaled_rouge_f1':

            def tokenize_text(text: str):
                return set(nltk_stopword_tokenize(text, 2, skip_unigrams=False))

            tokenized_sequence_1 = [tokenize_text(chunk.text) for chunk in sequence_1]
            tokenized_sequence_2 = [tokenize_text(turn.text) for turn in sequence_2]
            scoring_matrix = self.build_scoring_matrix(tokenized_sequence_1,
                                                       tokenized_sequence_2,
                                                       calculate_scaled_f1_turn_distribution)
        else:
            raise ValueError('Unexpected scoring method {}', scoring_method)

        raw_nw_result = self.needleman_wunsch_algorithm([chunk.id for chunk in sequence_1],
                                                        [chunk.id for chunk in sequence_2],
                                                        scoring_matrix)

        processed_nw_result = self._process_alignments(raw_nw_result,
                                                       sequence_1,
                                                       sequence_2)

        return processed_nw_result
