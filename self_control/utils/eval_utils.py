from transformers import RobertaForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from googleapiclient.errors import HttpError
from googleapiclient import discovery
from typing import Optional, Dict, List
import time

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

class PerspectiveApiScorer:
    """
    This class provides a method for accessing Perspective API using the Google API Client to obtain attribute scores for generated texts.
    """

    DEFAULT_ATTRIBUTES = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']

    def __init__(self, api_key=""):
        self.api_key = api_key

    def start(self):
        """
        :param api_key: the API key to use. For details, see https://support.perspectiveapi.com/s/docs-get-started
        """
        self._service = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def get_scores(self, input_text: str, requested_attributes: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get attribute scores for a given text via Perspective API.
        :param input_text: the input text
        :param requested_attributes: the attributes for which to compute scores
        :return: a mapping from attribute names to scores
        """
        requested_attributes = requested_attributes if requested_attributes else PerspectiveApiScorer.DEFAULT_ATTRIBUTES

        analyze_request = {
            'comment': {'text': input_text},
            'requestedAttributes': {attribute.upper(): {} for attribute in requested_attributes},
            'spanAnnotations': False,
            'languages': ['en'],
        }

        response = None
        while response is None:
            try:
                response = self._service.comments().analyze(body=analyze_request).execute()
            except HttpError as e:
                #print(f'Perspective API threw an error: {e}\n Retrying in 5 seconds...')
                #print(input_text)
                time.sleep(1)
                #return None

        return {attribute: response['attributeScores'][attribute.upper()]['summaryScore']['value'] for attribute in
                requested_attributes}