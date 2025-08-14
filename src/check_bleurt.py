import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

config = BleurtConfig.from_pretrained("lucadiliello/BLEURT-20")
model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20")
tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20")


def check_translation():

    references = ["Can it be delivered between 10 to 15 minutes?", "Soon the sun will set"]
    candidates = ["Can you send it for 10 to 15 minutes?", "Merhaba"]

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(references, candidates, padding="longest", return_tensors="pt")
        res = model(**inputs).logits.flatten().tolist()
    print(res)
    # [0.7885147929191589, 0.11762993037700653]


if __name__ == "__main__":
    check_translation()
