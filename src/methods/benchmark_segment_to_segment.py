import gc
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
from pathlib import Path
import subprocess
from time import sleep, time
from configuration import LABELS_SOURCE_PATH, PREDICTIONS_SOURCE_PATH, LANGUAGES_SHORT, LANGUAGES, PROMPTS, ROOT_PATH
from domain.TranslationTask import TranslationTask
from ollama import Client
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
import torch
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from bert_score import score

METHOD_NAME = "segment_to_segment"


def get_xcomet_xl_score(data: dict):
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    input_data = []

    for prediction in tqdm(data["paragraphs"]):
        input_data.append(
            {
                "src": prediction["main_language"],
                "mt": prediction["prediction"],
                "ref": prediction["other_language"],
            }
        )
    model_output = model.predict(input_data, batch_size=2, gpus=1)
    return round(model_output.system_score * 100, 2)


def get_wmt23_cometkiwi_da_xl_score(data: dict):
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)
    input_data = []

    for sample in tqdm(data["paragraphs"]):
        input_data.append(
            {
                "src": sample["main_language"],
                "mt": sample["prediction"],
            }
        )
    model_output = model.predict(input_data, batch_size=2, gpus=1)
    return round(model_output.system_score * 100, 2)


def get_bleurt_score(data: dict, batch_size: int = 50):
    model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20")
    tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20")

    references = [x["other_language"] for x in data["paragraphs"]]
    predictions = [x["prediction"] for x in data["paragraphs"]]
    scores = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(references), batch_size):
            refs_batch = references[i : i + batch_size]
            preds_batch = predictions[i : i + batch_size]
            inputs = tokenizer(
                refs_batch, preds_batch, padding="longest", max_length=512, truncation=True, return_tensors="pt"
            )
            res = model(**inputs).logits.flatten().tolist()
            scores.extend(res)
    return round(sum(scores) * 100 / len(scores), 2)


def get_bert_score(data: dict):
    references = [x["other_language"] for x in data["paragraphs"]]
    predictions = [x["prediction"] for x in data["paragraphs"]]
    precision, recall, f1 = score(predictions, references, model_type="bert-base-multilingual-cased", verbose=True)
    return round(f1.mean().item() * 100, 2)


def load_data(data_path: Path | str):
    data = json.loads(Path(data_path).read_text())
    return data


def get_content(translation_task: TranslationTask, prompt_name: str = "Prompt 3"):
    language_to_name = "English"
    languages_to = [x for x in LANGUAGES_SHORT if translation_task.language_to.lower()[:2] == x]

    if languages_to:
        language_to_name = LANGUAGES[LANGUAGES_SHORT.index(languages_to[0])]

    content = PROMPTS[prompt_name].format(language_to_name=language_to_name, text_to_translate=translation_task.text)
    return content


def get_prediction(model: str, text: str, language_from: str, language_to: str, prompt_name: str = "Prompt 3"):
    translation_task = TranslationTask(text=text, language_from=language_from, language_to=language_to)
    content = get_content(translation_task, prompt_name)
    response = Client().chat(model=model, messages=[{"role": "user", "content": content}])
    return response["message"]["content"]


def get_predictions_for_file(data: dict, model: str, prompt_name: str = "Prompt 3"):
    for paragraph in data["paragraphs"]:
        prediction = get_prediction(
            model, paragraph["main_language"], data["main_language"], data["other_language"], prompt_name
        )
        paragraph["prediction"] = prediction
    return data


def get_model_predictions(model: str, prompt_name: str = "Prompt 3"):
    predictions_path = Path(PREDICTIONS_SOURCE_PATH, METHOD_NAME, model)
    if not predictions_path.exists():
        predictions_path.mkdir(parents=True, exist_ok=True)

    # load model
    _ = Client().chat(model=model, messages=[{"role": "user", "content": "hello"}])

    model_translation_times_path = Path(ROOT_PATH, "results", "model_translation_times.csv")
    if not model_translation_times_path.exists():
        model_translation_times_path.write_text("model,method,prompt_name,file_name,sample_count,time\n")

    for label_data_path in sorted(LABELS_SOURCE_PATH.iterdir()):
        prediction_path = Path(predictions_path, label_data_path.name)
        if prediction_path.exists():
            print(f"Skipping {label_data_path.name}...")
            continue

        print(f"Processing {label_data_path.name}...")
        data = load_data(label_data_path)
        start_time = time()
        data = get_predictions_for_file(data, model)
        time_taken = time() - start_time
        prediction_path.write_text(json.dumps(data, indent=4))
        with open(model_translation_times_path, "a") as f:
            f.write(
                f"{model},{METHOD_NAME},{prompt_name},{label_data_path.name},{len(data['paragraphs'])},{round(time_taken, 2)}\n"
            )


def benchmark_model_translations(model: str, prompt_name: str = "Prompt 3"):
    predictions_path = Path(PREDICTIONS_SOURCE_PATH, METHOD_NAME, model)
    if not predictions_path.exists():
        print(f"Predictions for {model} not found")
        return

    subprocess.run(["ollama", "stop", model])
    sleep(1)

    benchmark_result_path = Path(ROOT_PATH, "results", "model_translation_benchmarks.csv")
    if not benchmark_result_path.exists():
        benchmark_result_path.write_text(
            "model,file_name,method,prompt_name,xcomet_xl,wmt23_cometkiwi_da_xl,bleurt,bert_score,average_score\n"
        )

    for prediction_path in sorted(predictions_path.iterdir()):
        if not prediction_path.exists():
            print(f"Prediction for {prediction_path.name} not found")
            continue

        print(f"Processing {prediction_path.name}...")

        data = load_data(prediction_path)

        print("Getting XCOMET-XL score...")
        xcomet_xl_score = get_xcomet_xl_score(data)
        print("Getting WMT23 COMETKIWI DA XL score...")
        wmt23_cometkiwi_da_xl_score = get_wmt23_cometkiwi_da_xl_score(data)
        print("Getting BLEURT score...")
        bleurt_score = get_bleurt_score(data)
        print("Getting BERT score...")
        bert_score = get_bert_score(data)

        print(f"XCOMET-XL score: {xcomet_xl_score}")
        print(f"WMT23 COMETKIWI DA XL score: {wmt23_cometkiwi_da_xl_score}")
        print(f"BLEURT score: {bleurt_score}")
        print(f"BERT score: {bert_score}")

        average_score = round((xcomet_xl_score + wmt23_cometkiwi_da_xl_score + bleurt_score + bert_score) / 4, 2)

        print(f"Average score: {average_score}")

        with open(benchmark_result_path, "a") as f:
            f.write(
                f"{model},{prediction_path.name},{METHOD_NAME},{prompt_name},{xcomet_xl_score},{wmt23_cometkiwi_da_xl_score},{bleurt_score},{bert_score},{average_score}\n"
            )

        torch.cuda.empty_cache()
        gc.collect()
        sleep(10)


if __name__ == "__main__":
    # get_model_predictions("gpt-oss")
    benchmark_model_translations("gpt-oss")
