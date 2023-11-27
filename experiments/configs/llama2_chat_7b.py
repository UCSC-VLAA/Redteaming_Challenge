import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_llama2_chat_7b'

    config.tokenizer_paths=["meta-llama/Llama-2-7b-chat-hf"]
    config.model_paths=["meta-llama/Llama-2-7b-chat-hf"]
    config.peft_dirs=[None]
    config.conversation_templates=['llama-2']

    return config