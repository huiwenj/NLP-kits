import evaluate
import numpy as np
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from evaluate import evaluator


classification_head_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_id = 0 if str(device) == 'cuda' else -1


# Step 1: Tokenize and Process the dataset tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(row):
  '''
  Pseudocode:
      1. Use the BERT tokenizer to tokenize the text in the "text" field of the input row.
      2. Ensure that the tokenized output is padded to the maximum length.
      3. Ensure that the tokenized output is truncated to the maximum length if it exceeds it.
      4. Return the tokenized output.

  Input:
      row: A dictionary representing a single row of the dataset.
            It contains at least the key "text" which holds a string of text.

  Returns:
      A dictionary with the tokenized output including keys "input_ids", "attention_mask", and "token_type_ids".
  '''
  return tokenizer(row['text'], padding='max_length', truncation=True, max_length=512)


def load_data():

    dataset = load_dataset("yelp_review_full")

    # Randomly select 1000 examples from the train and test data and tokenize
    train_data = dataset["train"].shuffle(seed=42).select(range(1000)).map(tokenize_function)
    eval_data = dataset["test"].shuffle(seed=42).select(range(1000)).map(tokenize_function)

    return train_data, eval_data


def train(train_data, eval_data):
    # Step 3: Specify the TrainingArguments and Initialize the Trainer
    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        output_dir="yelp-training",
        learning_rate=5e-05,
        num_train_epochs=3.0,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        full_determinism=True
    )

    trainer = Trainer(
        model=classification_head_bert,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model('./.models/yelpBERT')
    task_evaluator = evaluator("text_classification")

    yelpBERT = pipeline('text-classification', model='./.models/yelpBERT', tokenizer=tokenizer, device=device_id)

    eval_results = task_evaluator.compute(
        model_or_pipeline=yelpBERT,
        data=eval_data,
        metric=evaluate.load("accuracy"),
        label_mapping={"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2, "LABEL_3": 3, "LABEL_4": 4}
    )

    return eval_results


# Step 2: Define a compute_metrics function
def compute_metrics(eval_pred):
  '''
  Pseudocode:
      1. Extract the predicted probabilities and the true labels from the evaluation predictions.
      2. Compute the predicted labels by taking the argmax of the probabilities along the last axis.
      3. Calculate the accuracy by comparing the predicted labels with the true labels.
      4. Return a dictionary containing the accuracy.

  Input:
      eval_pred: A tuple (probabilities, labels)
                  probabilities: A 2D numpy array of shape (num_examples, num_classes) representing the predicted probabilities for each class.
                  labels: A 1D numpy array of shape (num_examples,) representing the true labels.

  Returns:
      A dictionary with the key "accuracy" and the value being the calculated accuracy.
  '''
  # Get the true labels and predicted probabilities
  probabilities, labels = eval_pred

  predictions = np.argmax(probabilities, axis=1)

  accuracy = np.sum(predictions == labels) / len(labels)

  return {'accuracy': accuracy}

classification_head_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)



