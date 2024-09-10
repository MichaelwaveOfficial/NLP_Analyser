from nltk.tokenize import sent_tokenize
from utils import load_subtitles_dataset
import spacy 
import pandas as pd
import os
from ast import literal_eval


class EntityClassifier(object):


    def __init__(self) -> None:
        self.nlp_model = self.load_model()
        pass 


    def load_model(self):
        nlp = spacy.load('en_core_web_trf')
        return nlp
    

    def infer_entity(self, script):

        # Store sentences and their entities. 
        model_output = []

        script_sentences = sent_tokenize(script)

        for sentence in script_sentences:

            document = self.nlp_model(sentence)

            # Store entities found per sentence. 
            entity_classes = set()

            for entity in document.ents:

                if entity.label_ == 'PERSON':
                    
                    fullname = entity.text
                    forename = fullname.split(' ')[0]
                    forename = forename.strip()
                    entity_classes.add(forename)

            model_output.append(entity_classes)

        return model_output
    

    def get_classes(self, dataset_path, save_path=None):

        ''' '''

        if save_path is not None and os.path.exists(save_path):
            dataframe = pd.read_csv(save_path)
            dataframe['entities'] = dataframe['entities'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return dataframe

        # Load dataset. 
        dataframe = load_subtitles_dataset(dataset_path=dataset_path)

        # Run model inference.
        dataframe['entities'] = dataframe['script'].apply(self.infer_entity)

        if save_path is not None:
            dataframe.to_csv(save_path)

        return dataframe


## 3:33