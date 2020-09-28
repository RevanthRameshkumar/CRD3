"""
This module is used to create the final TSV with the episode information. The output file is specified by
the TSV_OUTPUT_FILE. Make sure this is run only after the id's from bad_id.txt are resolved!
"""
import csv
from typing import Sequence

BASE_URL_FOR_TRANSCRIPT = 'https://criticalrole.fandom.com/wiki/'
EPISODE_AND_ID_FILENAME = 'episodes-and-ids.txt'
TSV_OUTPUT_FILE = 'output.tsv'
COLUMN_HEADINGS = [
    'episode name',
    'episode transcript url',
    'episode campaign number',
    'episode number in campaign',
    'episode id']


def get_transcript_url(episode_name: str) -> str:
    """
    returns the url for the transcript page based on episode name
    Args:
        episode_name: the episode's name

    Returns: the url for the transcript page based on episode name
    """
    return BASE_URL_FOR_TRANSCRIPT + \
        '{}/Transcript'.format(episode_name)


def write_to_tsv(output_file: str, line_to_write: Sequence):
    """
    appends a line to the tsv
    Args:
        output_file: the name of the file
        line_to_write: the line to write to the file
    """
    with open(output_file, 'a', newline='') as tsv_file:
        tsv_output = csv.writer(tsv_file, delimiter='\t')
        tsv_output.writerow(line_to_write)


def create_final_output_tsv():
    """
    Creates the final tsv (specified by the TSV_OUTPUT_FILE).
    Make sure to manually resolve all the ids from the bad_id list into the EPISODE_AND_ID_FILENAME file!
    """
    with open(EPISODE_AND_ID_FILENAME, 'r') as episode_and_id_input:
        episodes_and_ids = [line.strip()
                            for line in episode_and_id_input.readlines()]

    write_to_tsv(TSV_OUTPUT_FILE, COLUMN_HEADINGS)

    for line in episodes_and_ids:
        episode_id, episode_name, campaign_number, episode_number = line.split('$')
        line_to_write = [
            episode_name,
            get_transcript_url(episode_name),
            campaign_number,
            episode_number,
            episode_id,
        ]

        write_to_tsv(TSV_OUTPUT_FILE, line_to_write)


if __name__ == '__main__':
    create_final_output_tsv()
