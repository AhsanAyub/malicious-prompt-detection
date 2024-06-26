from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

class HFClassifier:
    def __init__(self, hf_path):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_path)
        self.classifier = pipeline(
          "text-classification",
          model=self.model,
          tokenizer=self.tokenizer,
          truncation=True,
          max_length=512,
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    
    def classify(self, text: str):
        return self.classifier(text)

def get_preds(hf_path, label="INJECTION"):
    cls = HFClassifier(hf_path=hf_path)
    
    # load dataset
    dataset = load_dataset("ahsanayub/malicious-prompts", split="test")
    df = dataset.to_pandas()
    # df = dataset.to_pandas()[:10]
    
    # get predictions
    predictions = []
    for text in tqdm(df.text):
        output = cls.classify(text)[0]
        # print(output)
        if output['label'] == label:
            predictions.append(output['score'])
        else:
            predictions.append(1-output['score'])
        
        
    return pd.DataFrame({'id': list(df['id']), 'y_pred': predictions})

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', help='Path to Hugging Face model', required=True)
parser.add_argument('-l','--label', help='output label', default="INJECTION")
parser.add_argument('-o','--output', help='Path to output file', required=True)
args = vars(parser.parse_args())

print(f"generating outputs from model {args['path']}")
output = get_preds(args['path'], label=args["label"])

print(f"saving outputs to {args['output']}")
output.to_csv(args['output'], index=False)