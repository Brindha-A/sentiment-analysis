import pickle
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


def predict(text):
    client = language.LanguageServiceClient()
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)
    score = annotations.document_sentiment.score
    print(score)
    if score<=-0.25:
        return 'Negative'
    elif score<=0.25:
        return 'Netural'
    else:
        return 'Positive'

def predictFromModel(text):
    vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))
    classifier = pickle.load(open('models/classifier.sav', 'rb'))
    text_vector = vectorizer.transform([text])
    result = classifier.predict(text_vector)
    return result[0]
