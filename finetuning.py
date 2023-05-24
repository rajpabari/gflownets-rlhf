from transformers import AutoModelForCausalLM, AutoTokenizer, DistilBertModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from llama import tokenizer

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model_id = "/home/ubuntu/gflownets-rlhf/download/7B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# NOTE: loading in 8bit, might be better to use float16 if it would improve performance
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", load_in_8bit=False
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

rewardModel = DistilBertModel()
rewardModel.load_state_dict("./saved-models/reward_model_final.pt")


# print(model.get_memory_footprint())
