
"""
This script demonstrates 
    - how to use a pretrained token classification model for Named Entity Recognition (NER) 
    using the Hugging Face Transformers library. 
    - It loads a specified model, tokenizes input text, and predicts entity labels for each token, 
    along with their confidence scores.
    Make sure to install the required libraries before running this script:
    conda create --file environment.yaml
    conda activate dutch_NER
"""
# Predict Using the pretrained model
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from torch.nn.functional import softmax
import os
import json


def ner(model_name, label_list, text, ner_output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define your labels 
    num_labels = len(label_list)
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in id2label.items()}

    # Load the token classification model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Set to evaluation mode
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def predict_tokens(text):
        """Predict token labels for input text"""
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            probabilities = softmax(outputs.logits[0], dim=-1)
        
        # Decode tokens and predictions
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        results = []
        
        for token, pred_id, probs in zip(tokens, predictions, probabilities):
            pred_label = id2label[pred_id.item()]
            max_prob = torch.max(probs).item()
            results.append((token, pred_label, max_prob))
        
        return results

    predictions = predict_tokens(text)

    valid_predictions = []

    for _, (token, label, confidence) in enumerate(predictions):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            valid_predictions.append({
            "token": token,
            "label": label, 
            "confidence": round(float(confidence), 3)
        })
    filename = f"{os.path.join(ner_output_dir, 'ner_output')}.json"

    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(valid_predictions, f, indent=2, ensure_ascii=False)
  
    # Pretty print results
    
if "__main__" == __name__:
    model_name = "emanjavacas/GysBERT"
    label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    text= "Jan de Vries werkt bij de Universiteit van Amsterdam in Nederland."
    output_dir = os.getcwd()
    ner(model_name=model_name, label_list=label_list, text=text, ner_output_dir=output_dir)
