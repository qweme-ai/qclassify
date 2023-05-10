# QClassifyPipeline
A single file pipeline for sequence classification

### Usage
```py
from qclassify import QClassifyPipeline

device = "cpu"
labels = ["neutral", "porn", "cp", "violence"]
model_name = "Your model here"
pipe = QClassifyPipeline(model_name, labels, device)

def main() -> int:
    try:
        while True:
            text = input("> ")
            print(pipe(text).to_dict())
        
    except KeyboardInterrupt:
        return 0
    
if __name__ == "__main__":
    exit(main())
```
```
> naked anime girl 
```
```json
{
    'labels': [
        {
            'label': 'neutral', 
            'score': 0.002601000713184476
        }, 
        {
            'label': 'porn', 
            'score': 0.9957257509231567
        }, 
        {
            'label': 'cp', 
            'score': 0.001233138027600944
        }, 
        {
            'label': 'violence', 
            'score': 0.0004401823098305613
        }
    ], 
    'score': 0.9957257509231567, 
    'prediction': 'porn'
}
```

