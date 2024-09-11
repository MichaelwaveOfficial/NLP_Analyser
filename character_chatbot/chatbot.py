import torch 
import huggingface_hub
import pandas as pd 
import re 
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import gc


class Chatbot(object):


    def __init__(self, model_path, data_path='/content/data/natuto.csv', huggingface_token=None) -> None:
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
            print('Model not found in huggingface hub, model self training.')
            
            # load dataset.
            training_data = self.load_data()

            # train model
            self.train_model(self.base_model_path, training_data)

            # load model. 
            self.model = self.load_model(self.model_path)

    
    def load_data(self):
        transcript = pd.read_csv(self.data_path)
        transcript = transcript.dropna()
        transcript['line'] = transcript['line'].apply(remove_parenthesis)
        transcript['no_of_words'] = transcript['line'].str.strip().str.split(' ')
        transcript['no_of_words'] = transcript['no_of_words'].apply(lambda x: len(x))
        transcript['naruto_response_flag'] = 0 
        transcript.loc[ (transcript['name']=='Naruto')&(transcript['no_of_words']>5)]=1

        indicies = list(transcript[(transcript['naruto_response_flag']==1)&(transcript.index>0)].index)

        ## System Prompt
        system_prompt = ''' Your role is to assimilate the role of the character named "Naruto" from the renowned anime serioes "Naruto".
        Your responses, when prompted, should reflect that of his personality and his speech patterns to best imitate his likeliness.\n '''

        prompts = []

        for index in indicies:

            prompt = system_prompt

            prompt += transcript.iloc[index -1]['line']
            prompt += '\n'
            prompt += transcript.iloc[index]['line']

            prompts.append(prompt)

        
        dataframe = pd.DataFrame({'prompt': prompts})

        dataset = Dataset.from_pandas(dataframe)

        return dataset
    

    def train_model(self, dataset, model_path, output_dir='./results', train_batch_size=4, optimiser='paged_adamw_32bit', save_steps=200, logging_steps=10, learning_rate=2e-4, max_gradient=0.3, max_steps=300, warmup_ratio=0.3, lr_scheduler='constant'):
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)

        model = AutoModelForCausalLM.from_pretrained(model_path, quntization_config=bnb_config, trust_remote_code=True)

        model.config.use_cache = False 

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        lora_alpha = 16 
        lora_dropout = 0.1 
        lora_r = 64

        peft_config = LoraConfig(lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_r=lora_r, bias='none', task_type='CASUAL_LM')

        training_arguments = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=1,
            optim=optimiser,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            fp16=True,
            max_grad_norm=max_gradient, 
            max_steps=max_steps, 
            warmup_ratio=warmup_ratio, 
            group_by_length=True, 
            lr_scheduler_type=lr_scheduler,
            report_to='none'
        )

        max_sequence_len = 512 

        trainer = SFTTrainer(
            model=model, 
            train_dataset=dataset, 
            peft_config=peft_config, 
            dataset_text_field='prompt', 
            max_seq_length=max_sequence_len, 
            tokenizer=tokenizer, 
            args=training_arguments
        )

        trainer.train()

        trainer.model.save_pretrained('final_ckpt')
        tokenizer.save_pretrained('final_ckpt')

        del trainer, model
        gc.collect()

        base_model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True, quanitization_config=bnb_config, torch_dtype=torch.float16, device_map=self.device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = PeftModel.from_pretrained(base_model, 'final_ckpt')
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        del model, base_model
        gc.collect()


    def load_model(self, model_path):

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)

        pipeline = pipeline('text-generation', model=model_path, model_kwargs={'torch_dtype': torch.float16, 'quantization_config' : bnb_config})

        return pipeline


    def chat(self, message, history):

        messages = []

        messages.append(
            ''' Your role is to assimilate the role of the character named "Naruto" from the renowned anime serioes "Naruto".
                 Your responses, when prompted, should reflect that of his personality and his speech patterns to best imitate his likeliness.\n '''
        )

        for msgs in messages:
            messages.append({ 'role' : 'user', 'content' : msgs[0] })
            messages.append({ 'role' : 'assistant', 'content' : msgs[1] })

        messages.append({'role' : 'user' , 'content' : message })

        terminator = [self.model.tokenizer.eos_token_id, self.model.tokenizer.convert_tokens_to_ids('<|eot_id|>')]

        output = self.model(messages, max_length=256, eos_token_id=terminator, do_sample=True, temperature=0.6, top_p=0.9)

        return output[0]['generated_text'][-1]


def remove_parenthesis(text):
    return re.sub(r'\(.*?\)', '', text)