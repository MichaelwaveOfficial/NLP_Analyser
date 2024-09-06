from transformers import pipeline
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import torch 
import os 
import sys 
import pathlib
import nltk

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset

nltk.download('punkt')
nltk.download('punkt_tab')


class ThemeClassifier(object):


    def __init__(self, theme_list) -> None:
        self.model_name = 'facebook/bart-large-mnli'
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)


    def load_model(self, device):

        ''' '''

        theme_classifier = pipeline(
            'zero-shot-classification',
            model=self.model_name, 
            device=device
        )

        return theme_classifier
    

    def get_themes_inference(self, script):

        ''' '''

        batch_size = 20
        script_batches = []

        ''' Process the sentences into batches. '''

        script_sentences = sent_tokenize(script)

        for index in range(0, len(script_sentences),batch_size):

            sentence = ' '.join(script_sentences[index:index+batch_size])
            script_batches.append(sentence)
        
        ''' Run model on the processed, batched data. '''

        theme_output = self.theme_classifier(
            script_batches, 
            self.theme_list, 
            multi_label=True
        )

        ''' Create meaningful output. '''

        themes = {}

        for output in theme_output:

            for label, score in zip(output['labels'], output['scores']):

                if label not in themes:
                    themes[label] = []
                themes[label].append(score)

        mean_themes = { key: np.mean(np.array(theme)) for key, theme in themes.items() }

        return mean_themes
    

    def get_themes(self, dataset_path, save_path=None):

        ''' Load data from set path, save output to stub to mitigate model exhaustion. '''

        # If path exists, read preexisting data into application.
        if save_path is not None and os.path.exists(save_path):
            dataset_df = pd.read_csv(save_path)
            return dataset_df

        # Load dataset. 
        dataset_df = load_subtitles_dataset(dataset_path=dataset_path)

        # Run model inference.
        output_themes = dataset_df['script'].apply(self.get_themes_inference)

        themes_df = pd.DataFrame(output_themes.tolist())
        dataset_df[themes_df.columns] = themes_df 

        # Save output.
        if save_path is not None:
            dataset_df.to_csv(save_path, index=False)

        return dataset_df