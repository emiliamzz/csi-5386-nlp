from pathlib import Path
from fire import Fire

from run_inference import run_inference
from voting_ensemble import voting_ensemble

# ここは入れたモデル名に応じて変えてください
MODEL_PATHS = ['/workspace/model/seed43','/workspace/model/seed45','/workspace/model/seed46','/workspace/model/seed47','/workspace/model/seed48']

def run_wrapper(data_path : str, out_path : str) -> None:
    tmp_out_dir = '/workspace/tmp_json'
    
    for i,model_path in enumerate(MODEL_PATHS):
        print(f'start inference : {model_path}')
        run_inference(model_path,data_path,tmp_out_dir+f"/{i}.jsonl")
        print(f'done')
    
    voting_ensemble(data_path,tmp_out_dir,out_path)


if __name__ == '__main__':
    Fire(run_wrapper)