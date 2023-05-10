from typing import List
from torch import no_grad
from torch.nn import functional as F
from torch.cuda import empty_cache
from transformers import \
    AutoModelForSequenceClassification, \
    AutoTokenizer


class Response:
    def __init__(self, p: str, pb: List[float], l: List[str]) -> None:
        self.prediction = p
        self.prediction_label = l[p]
        self.probabilities = pb
        self.labels = l

    def to_dict(self) -> dict:
        out = {}
        out["labels"] = []

        index = 0
        for label in self.labels:
            if label == self.prediction_label:
                out["score"] = self.probabilities[index]
            
            out["labels"].append({"label": label, "score": self.probabilities[index]})
            index += 1

        out["prediction"] = self.prediction_label

        return out


class QClassifyPipeline:
    def __init__(self, model: str, labels: List[str], device: str) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.labels = labels
        self.device = device

    def classify(self, text: str) -> Response:
        if "cuda" in self.device:
            empty_cache()
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with no_grad():
            logits = self.model(**inputs).logits
        
        probabilities = F.softmax(logits, dim=-1)[0].tolist()
        prediction = logits.argmax(dim=-1).item()

        return Response(prediction, probabilities, self.labels)
    
    def __call__(self, text: str) -> Response:
        return self.classify(text)