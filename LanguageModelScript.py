
get_ipython().system('pip install git+https://github.com/huggingface/transformers')
get_ipython().system("pip list | grep -E 'transformers|tokenizers'")

get_ipython().system('mkdir HindiBERTo')
tokenizer.save_model("HindiBERTo")

import json
config = {
"architectures": [
"RobertaForMaskedLM"
],
"attention_probs_dropout_prob": 0.1,
"hidden_act": "gelu",
"hidden_dropout_prob": 0.1,
"hidden_size": 768,    
"initializer_range": 0.02,
"intermediate_size": 3072,
"layer_norm_eps": 1e-05,
"max_position_embeddings": 514,
"model_type": "roberta",
"num_attention_heads": 12,
"num_hidden_layers": 12,
"type_vocab_size": 1,
"vocab_size": 50265
}

with open("HindiBERTo/config.json", 'w') as fp:
    json.dump(config, fp)

tokenizer_config = {"max_len": 512}

with open("HindiBERTo/tokenizer_config.json", 'w') as fp:
    json.dump(tokenizer_config, fp)

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./HindiBERTo/vocab.json",
    "./HindiBERTo/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)



tokens = (tokenizer.encode("इंद्रधनुष में 7 रंग होते हैं"))

print(tokens)

import torch
torch.cuda.is_available()

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)


from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./HindiBERTo", max_len=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

model.num_parameters()
# => 84 million parameters

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./hin_wikipedia_2021_30K-words.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./HindiBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()


trainer.save_model("./HindiBERTo")

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./HindiBERTo",
    tokenizer="./HindiBERTo"
)


fill_mask("इंद्रधनुष में 7 रंग होते <mask>")

