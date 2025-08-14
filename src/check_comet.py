import os
from comet import download_model, load_from_checkpoint
from comet.models.base import Prediction
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# model_path = download_model("Unbabel/XCOMET-XL") # reference-based
# or for example:
# model_path = download_model("Unbabel/wmt22-comet-da")
# model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")  # reference-free
# model_path = download_model("Unbabel/unite-mup")

print(model_path)
# Load the model checkpoint:
model = load_from_checkpoint(model_path)


def check_translation():
    data = [
        {
            "src": "10 到 15 分钟可以送到吗",
            "mt": "Can I receive my food in 10 to 15 minutes?",
            # "mt": "Can it be delivered between 10 to 15 minutes?",
            # "ref": "Can it be delivered between 10 to 15 minutes?",
        },
        {
            "src": "Pode ser entregue dentro de 10 a 15 minutos?",
            "mt": "Can you send it for 10 to 15 minutes?",
            # "mt": "Can it be delivered between 10 to 15 minutes?",
            # "ref": "Can it be delivered between 10 to 15 minutes?",
        },
        {
            "src": "Soon the sun will set",
            # "src": "Pronto se pondrá el sol",
            # "ref": "Soon the sun will set",
            # "ref": "Soon the sun will set",
            "mt": "Merhaba",
        },
        {
            "src": "Soon the sun will set",
            # "src": "Pronto se pondrá el sol",
            # "ref": "Soon the sun will set",
            # "ref": "Soon the sun will set",
            "mt": "fdggfdgdf",
        },
    ]
    model_output: Prediction = model.predict(data, batch_size=8, gpus=1)
    print(model_output.scores)
    print(model_output.system_score)
    # print(model_output.metadata.error_spans)


if __name__ == "__main__":
    check_translation()
