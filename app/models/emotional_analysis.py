from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

classification_head_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

