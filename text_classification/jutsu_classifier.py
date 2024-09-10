import pandas as pd
import torch
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, pipeline
from .cleaner import DataCleaner
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
from .training_utils import get_class_weights
from .trainer import Trainer
from .training_utils import compute_metrics
import gc


class JutsuClassifier(object):


    def __init__(self, model_path, data_path=None, column_name = 'text', label_name = 'jutsu', model_name = 'distilbert/distilbert-base-uncased', test_size=0.2, label_count=3, huggingface_token=None) -> None:

        self.model_path        =        model_path
        self.data_path         =         data_path
        self.column_name       =       column_name
        self.label_name        =        label_name 
        self.model_name        =        model_name
        self.test_size         =         test_size
        self.label_count       =       label_count
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.huggingface_token = huggingface_token

        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        self.tokenizer = self.load_tokenizer()

        if not huggingface_hub.repo_exists(self.model_path):

            # Check path is provided. 
            if data_path is None:
                raise ValueError('Data path required for model training, path does not exist in huggingface hub.')
            
            train_data, test_data = self.load_data(self.data_path)

            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()

            data_aggregation = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(data_aggregation)

            self.train_model(train_data, test_data, class_weights)

        self.model = self.load_model(self.model_path)

    
    def load_model(self, model_path):

        model = pipeline('text-classification', model=model_path, top_k=1, device=self.device)
        return model
    
    
    def train_model(self, train_data, test_data, class_weights):

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.label_count, id2label=self.label_dict)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_arguments = TrainingArguments(
            output_dir = self.model_path, 
            learning_rate=2e-4, 
            per_gpu_train_batch_size=8, 
            per_gpu_eval_batch_size=8, 
            num_train_epochs=5, 
            weight_decay=0.01, 
            evaluation_strategy='epoch', 
            logging_dir='epoch', 
            push_to_hub=True
        )

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_data,
            eval_dataset = test_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.set_device(device=self.device)
        trainer.set_class_weights(class_weights=class_weights)

        trainer.train()

        del trainer, model
        gc.collect()

        if self.device == 'cuda':
            torch.cuda.empty_cache()


    def load_tokenizer(self):
 
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer
  
    
    def load_data(self, data_path):

        test_size = 0.2

        # Process data. 
        data = pd.read_json(data_path, lines=True)
        data['simplified_jutsu'] = data['jutsu_type'].apply(self.simplify_jutsus)
        data['text'] = data['jutsu_name'] + '. ' + data['jutsu_desc']
        data[self.label_name] = data['simplified_jutsu']
        data = data[['text', self.label_name]]
        data = data.dropna()

        # Clean Data. 
        data_cleaner = DataCleaner()
        data['clean_text'] = data[self.column_name].apply(data_cleaner.clean_text)

        # Encode data labels. 
        lbl_encoder = preprocessing.LabelEncoder()
        lbl_encoder.fit(data[self.label_name].tolist())

        label_dict = { index:label for index, label in enumerate(lbl_encoder.__dict__['classes_'].tolist()) }
        self.label_dict = label_dict
        data['label'] = lbl_encoder.transform(data[self.label_name].tolist())

        # Train / Test data split for model training ++ validation. 
        training_data, testing_data = train_test_split(data, test_size=test_size, stratify=data['label'])

        # Pandas format -> hugging face format. 
        training_set = Dataset.from_pandas(training_data)
        testing_set = Dataset.from_pandas(testing_data)

        # Tokenize data for model. 
        tokenized_training_set = training_set.map(lambda examples: self.preprocess_sets(self.tokenizer, examples), batched=True)
        tokenized_testing_set = testing_set.map(lambda examples: self.preprocess_sets(self.tokenizer, examples), batched=True)

        return tokenized_training_set, tokenized_testing_set


    def simplify_jutsus(self, jutsu):
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"
        
    
    def preprocess_sets(self, tokenizer, examples):
        return tokenizer(examples['clean_text'], truncation=True)
    

    def postprocess(self,model_output):
        output=[]
        for pred in model_output:
            label = max(pred, key=lambda x: x['score'])['label']
            output.append(label)
        return output


    def classify_jutsu(self, text):
        model_output = self.model(text)
        predictions = self.postprocess(model_output) 

        return predictions
