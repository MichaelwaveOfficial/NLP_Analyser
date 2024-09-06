from glob import glob 
import pandas as pd 


def load_subtitles_dataset(dataset_path):

    '''
        Load dataset and clean directory filenames to extract meaningful data. 
    '''

    scripts = []
    episodes = []

    subs_path = glob(dataset_path + '/*.ass')

    for path in subs_path:

        with open(path, 'r', encoding='utf-8') as sub_file:
            lines = sub_file.readlines()
            lines = lines[27:]
            lines = [','.join(line.split(',')[9:]) for line in lines]

        lines = [ line.replace('\\N', ' ')  for line in lines ]
        script = ' '.join(lines)

        episode_no = int(path.split('-')[-1].split('.')[0].strip())

        scripts.append(script)
        episodes.append(episode_no)

    episodes_df = pd.DataFrame.from_dict({'episode:':episodes, 'script':scripts})

    return episodes_df
    