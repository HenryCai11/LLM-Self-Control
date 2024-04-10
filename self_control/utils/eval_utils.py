from transformers import RobertaForSequenceClassification, AutoTokenizer
from scipy.special import softmax

emotion_eval_model = RobertaForSequenceClassification.from_pretrained("/home/models/twitter-roberta-base-sentiment-latest/")
emotion_tokenizer = AutoTokenizer.from_pretrained("/home/models/twitter-roberta-base-sentiment-latest/")

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def test_emotion(input_text, emotion_eval_model=emotion_eval_model, emotion_tokenizer=emotion_tokenizer):
    verbalization = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    input_text = preprocess(input_text)
    encoded_input = emotion_tokenizer(input_text, return_tensors='pt')
    encoded_input["input_ids"] = encoded_input["input_ids"][:, :512]
    encoded_input["attention_mask"] = encoded_input["attention_mask"][:, :512]
    output = emotion_eval_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return verbalization[scores.argmax(axis=-1)], scores