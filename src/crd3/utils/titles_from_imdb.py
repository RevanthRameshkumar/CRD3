"""
Populates a file specified by EPISODE_NAME_AND_ID_FILENAME with episode ID's and BAD_IDS_FILENAME with episode names
needing further review. Once the BAD_IDS_FILENAME is fully reviewed, move on to the second pass. The ID's are obtained
using the IMDB api to get the names and the Wiki api to get the internal page IDs.
"""
from imdb import IMDb
import time
import requests
from typing import TextIO, List

IMDB_ID_FOR_CRITICAL_ROLE = '4834232'
WIKI_BASE_URL = 'https://criticalrole.fandom.com/api/v1/Articles/'
EPISODE_ID_ENDPOINT = 'List?offset='
ID_TEST_ENDPOINT = 'AsSimpleJson?id='

EPISODE_NAME_AND_ID_FILENAME = 'episodes-and-ids.txt'
BAD_IDS_FILENAME = 'bad_ids.txt'


def extract_episode_info_from_season(season: List[str], season_number: int) -> List:
    """

    Args:
        season: list of episode names from the IMDB api
        season_number: season (should be same as campaign number)

    Returns: a List containing episode info for the specified season (containing cleaned episode name, campaign number,
        and episode number)

        [
            {
                episode_name:,
                campaign_number:,
                episode_number:
            },
            ...
        ]

    """
    episode_info_list = []
    for i in range(1, len(season)):
        episode_name = str(season[i])
        cleaned_name = episode_name.replace(' ', '_').replace('...', '')
        episode_info_list.append({
            'episode_name': cleaned_name,
            'campaign_number': season_number,
            'episode_number': i,
        })
    return episode_info_list


def get_id(episode_name: str) -> str:
    """
    Queries the wiki API to get the episode ID based on the episode name.
    Args:
        episode_name: the name of the Critical Role episode

    Returns: the episode ID from wiki api

    """
    # throwing this in so we don't hammer the API
    time.sleep(3)
    print('querying for episode: {}'.format(episode_name))
    r = requests.get(WIKI_BASE_URL + EPISODE_ID_ENDPOINT + episode_name)
    if r.status_code == 200:
        data = r.json()
        return str(data['items'][0]['id'])


def get_text(json_data: dict) -> str:
    """
    Args:
        json_data: the json data returned from querying the IMDB api w/ episode id

    Returns: text from json content (under sections->content->text). Returns None if there is an exception.

    """
    try:
        return json_data['sections'][0]['content'][0]['text']
    except (KeyError, IndexError):
        pass


def test_id(episode_id: str) -> str:
    """
    tests if the episode id by checking if there is content using wiki api
    Args:
        episode_id: the episode id to test

    Returns: a 'needs review' string if there is an issue, otherwise None

    """
    # another attempt to be good API users
    time.sleep(3)
    print('querying for id: {}'.format(episode_id))
    r = requests.get(WIKI_BASE_URL + ID_TEST_ENDPOINT + episode_id)
    result = r.json()
    if get_text(result) is None:
        return 'episode {} needs review'.format(episode_id)


def write_to_bad_id_file(bad_id: str):
    """
    writes a bad_id to the BAD_IDS_FILENAME file
    Args:
        bad_id: the bad id to record

    """
    with open(BAD_IDS_FILENAME, 'a') as bad_id_file:
        bad_id_file.write(bad_id)
        bad_id_file.write('\n')


def write_episode_data(
                       episodes_and_ids_file: TextIO,
                       episode_id: str,
                       episode_name: str,
                       campaign_number: int,
                       episode_number: int):
    """

    Args:
        episodes_and_ids_file: file to write into
        episode_id: episode id (the wiki internal one)
        episode_name: name of critical role episode
        campaign_number: campaign to which the episode belongs
        episode_number: number of episode in the campaign

    """
    episodes_and_ids_file.write(
        '{}${}${}${}'.format(
            episode_id,
            episode_name,
            campaign_number,
            episode_number))
    episodes_and_ids_file.write('\n')


def write_episode_ids():
    """
    Writes episode id's into the EPISODE_NAME_AND_ID_FILENAME and episodes needing review into the BAD_IDS_FILENAME.
    Uses IMDB as source of truth for episode names, which ae then used to get ids.
    """
    ia = IMDb()
    # for some bizarre reason, the imdb api treats everything as a movie
    series = ia.get_movie(IMDB_ID_FOR_CRITICAL_ROLE)
    ia.update(series, 'episodes')
    season1 = series['episodes'][1]
    season2 = series['episodes'][2]

    all_episodes = extract_episode_info_from_season(
        season1, 1) + extract_episode_info_from_season(season2, 2)

    with open(EPISODE_NAME_AND_ID_FILENAME, 'w') as episodes_and_ids_file:
        for item in all_episodes:
            episode_id = get_id(item['episode_name'])
            bad_id = test_id(episode_id)
            if bad_id is not None:
                # we need a list of these to manually review the full episode-id list
                write_to_bad_id_file(bad_id)
            # this is the full list so we can manually review and update it
            write_episode_data(
                episodes_and_ids_file,
                episode_id,
                item['episode_name'],
                item['campaign_number'],
                item['episode_number'])


if __name__ == '__main__':
    write_episode_ids()
