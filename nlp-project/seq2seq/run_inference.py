from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          default_data_collator, set_seed)
from tqdm import tqdm
import json
from fire import Fire
import torch

from datasets import load_dataset
DEVICE='cuda'


def run_inference(model_path : str, data_path: str , out_path : str, max_length: int = 2048):
    dataset = load_dataset('/workspace/seq2seq/load_sqlike_dataset_both.py', data_dir = data_path)
    val_dataset = dataset['validation']
    with torch.no_grad():
        # model_path = 'google/flan-t5-large'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path,n_positions = max_length).to(DEVICE)
        model.eval()
        def get_answer(question, context):
            input_text = "question : %s  context : %s" % (question, context)

            with torch.no_grad():
                features = tokenizer([input_text], return_tensors='pt', truncation = True, max_length=max_length).to(DEVICE)
                out = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],max_length=max_length)
                answer = tokenizer.decode(out[0],skip_special_tokens=True)
                del features
            return answer

        results = []
        for i in tqdm(range(len(val_dataset))):
            uuid = val_dataset[i]['id']
            question,context = val_dataset[i]['question'],val_dataset[i]['context']
            answer = get_answer(question,context)
            results.append(
                {'uuid':uuid,'spoiler':answer}
            )
            del answer

    with open(out_path,'w') as f:
        for result in results:
            f.write(json.dumps(result)+'\n')


if __name__ == '__main__':
    Fire(run_inference)