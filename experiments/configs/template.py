from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # General parameters 
    config.target_weight=1.0
    config.control_weight=0.0
    config.progressive_goals=False
    config.progressive_models=False
    config.anneal=False
    config.stop_on_success=True
    config.early_stop=False
    config.verbose=False
    config.allow_non_ascii=False
    config.num_train_models=1
    config.change_prefix_step=3
    config.warm_up=0.0
    config.choose_control_prob=0.5
    config.enable_prefix_sharing=True

    # Results
    config.result_prefix = None

    # tokenizers
    config.tokenizer_paths=None
    config.tokenizer_kwargs=[{"use_fast": True}]
    
    config.model_paths=None
    config.model_kwargs=[{"low_cpu_mem_usage": True, "use_cache": False, "use_flash_attention_2": True}]
    config.conversation_templates=None
    config.devices=['cuda:0']

    # data
    config.train_data = ''
    config.test_data = ''
    config.n_train_data = 50
    config.n_test_data = 0
    config.data_offset = 0

    # attack-related parameters
    config.attack = 'gcg'
    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.n_steps = 150
    config.test_steps = 10
    config.batch_size = 128
    config.topk = 64
    config.temp = 1
    config.filter_cand = True
    config.model = ""
    config.setup = ""

    return config
