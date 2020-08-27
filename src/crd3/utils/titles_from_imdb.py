from imdb import IMDb
import time
import requests

IMDB_ID_FOR_CRITICAL_ROLE = '4834232'
WIKI_BASE_URL = 'https://criticalrole.fandom.com/api/v1/Articles/'
EPISODE_ID_ENDPOINT = 'List?offset='
ID_TEST_ENDPOINT = 'AsSimpleJson?id='

EPISODE_NAME_AND_ID_FILENAME = 'episodes-and-ids.txt'
BAD_IDS_FILENAME = 'bad_ids.txt'

ia = IMDb()

# for some bizarre reason, the imdb api treats everything as a movie
series = ia.get_movie(IMDB_ID_FOR_CRITICAL_ROLE)

ia.update(series, 'episodes')

season1 = series['episodes'][1]
season2 = series['episodes'][2]


def extract_episode_info_from_season(season, season_number):
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


def get_id(episode_name):
    # throwing this in so we don't hammer the API
    time.sleep(3)
    print('querying for episode: {}'.format(episode_name))
    r = requests.get(WIKI_BASE_URL + EPISODE_ID_ENDPOINT + episode_name)
    if r.status_code == 200:
        data = r.json()
        return str(data['items'][0]['id'])


def get_text(json_data):
    try:
        return json_data['sections'][0]['content'][0]['text']
    except BaseException:
        pass


def test_id(episode_id):
    # another attempt to be good API users
    time.sleep(3)
    print('querying for id: {}'.format(episode_id))
    r = requests.get(WIKI_BASE_URL + ID_TEST_ENDPOINT + episode_id)
    result = r.json()
    if get_text(result) is None:
        return 'episode {} needs review'.format(episode_id)


def write_to_bad_id_file(bad_id):
    with open(BAD_IDS_FILENAME, 'a') as bad_id_file:
        bad_id_file.write(bad_id)
        bad_id_file.write('\n')


def write_episode_data(
        episode_id,
        episode_name,
        campaign_number,
        episode_number):
    with open(EPISODE_NAME_AND_ID_FILENAME, 'a') as episodes_and_ids_file:
        episodes_and_ids_file.write(
            '{}${}${}${}'.format(
                episode_id,
                episode_name,
                campaign_number,
                episode_number))
        episodes_and_ids_file.write('\n')


all_episodes = extract_episode_info_from_season(
    season1, 1) + extract_episode_info_from_season(season2, 2)

for item in all_episodes:
    episode_id = get_id(item['episode_name'])
    bad_id = test_id(episode_id)
    if bad_id is not None:
        # we need a list of these to manually review the full episode-id list
        write_to_bad_id_file(bad_id)
    # this is the full list so we can manually review and update it
    write_episode_data(
        episode_id,
        item['episode_name'],
        item['campaign_number'],
        item['episode_number'])
