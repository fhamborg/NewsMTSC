import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from NewsSentiment import TargetSentimentClassifier

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
TEXT = "On some great and glorious day the plain folks of the land will reach their heart's desire at last, and the White House will be adorned by a downright moron."
ner_spans = nlp(TEXT)
ents = [span["word"] for span in ner_spans]
print(f"Entities: {ents}")
tsc = TargetSentimentClassifier()
for span in ner_spans:
    l = TEXT[:span['start']]
    m = TEXT[span['start']:span['end']]
    r = TEXT[span['end']:]
    sentiment = tsc.infer_from_text(l, m, r)
    print(f"{span['entity']}\t{sentiment[0]['class_label']}\t{sentiment[0]['class_prob']:.2f}\t{m}")
    
