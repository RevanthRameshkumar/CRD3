import csv

BASE_URL_FOR_TRANSCRIPT = 'https://criticalrole.fandom.com/wiki/'
EPISODE_AND_ID_FILENAME = 'episodes-and-ids.txt'
TSV_OUTPUT_FILE = 'output.tsv'
COLUMN_HEADINGS = [
    'episode name',
    'episode transcript url',
    'episode campaign number',
    'episode number in campaign',
    'episode id']


def get_transcript_url(episode_name):
    return BASE_URL_FOR_TRANSCRIPT + \
        '{}/Transcript'.format(episode_name)


def write_to_tsv(output_file, line_to_write):
    with open(output_file, 'a', newline='') as tsv_file:
        tsv_output = csv.writer(tsv_file, delimiter='\t')
        tsv_output.writerow(line_to_write)


# after manually correcting the ids from the bad_id list, we're ready to
# create the final tsv
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
