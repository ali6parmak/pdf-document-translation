import gc
import json
import os
import subprocess
import torch
from configuration import PROMPTS

from pathlib import Path
from time import time, sleep

import pandas as pd
from shutil import rmtree
from os.path import join

from tqdm import tqdm
from huggingface_hub import hf_hub_download

from configuration import ROOT_PATH, LANGUAGES_SHORT, LANGUAGES
from domain.TranslationTask import TranslationTask
from fast_bleu import BLEU

from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from bert_score import score

from comet import download_model, load_from_checkpoint
from huggingface_hub import login
from dotenv import load_dotenv
from ollama import Client

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))


def get_content(translation_task: TranslationTask, prompt_name: str = "Prompt 3"):
    language_to_name = "English"
    languages_to = [x for x in LANGUAGES_SHORT if translation_task.language_to.lower()[:2] == x]

    if languages_to:
        language_to_name = LANGUAGES[LANGUAGES_SHORT.index(languages_to[0])]

    content = PROMPTS[prompt_name].format(language_to_name=language_to_name, text_to_translate=translation_task.text)
    return content


def download_data(language_pairs: list[str] = ["en-ru"]):
    repo_id = "Helsinki-NLP/opus-100"
    train_file_name = "test-00000-of-00001.parquet"
    for pair in language_pairs:
        file_name = join(pair, train_file_name)
        hf_hub_download(
            repo_id=repo_id, filename=file_name, local_dir=join(f"{ROOT_PATH}/language_data"), repo_type="dataset"
        )
    rmtree(join(ROOT_PATH, "language_data", ".huggingface"))


def read_samples(language_pair: str, limit: int = 0) -> list[tuple[str, str]]:
    df = pd.read_parquet(join(ROOT_PATH, "language_data", language_pair), engine="pyarrow")
    lang1, lang2 = df.iloc[0]["translation"].keys()
    texts_translations = list()
    for i, row in df.iterrows():
        texts_translations.append((row["translation"][lang1], row["translation"][lang2]))
        if limit and i == limit:
            break

    return texts_translations


def get_bleu_score(samples: list[tuple[str, str]], predictions: list[str]):

    total_score = 0
    for i, (source_text, reference_text) in tqdm(enumerate(samples)):
        prediction = predictions[i].replace("```", "")
        list_of_references = [reference_text.split()]
        hypotheses = [prediction.split()]
        weights = {"bigram": (1 / 2.0, 1 / 2.0), "trigram": (1 / 3.0, 1 / 3.0, 1 / 3.0)}
        bleu = BLEU(list_of_references, weights)
        total_score += (bleu.get_score(hypotheses)["bigram"][0] + bleu.get_score(hypotheses)["trigram"][0]) / 2.0
    return round(total_score * 100 / len(samples), 2)


def get_xcomet_xl_score(samples: list[tuple[str, str]], predictions: list[str]):
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    data = []

    for i, (source_text, reference_text) in tqdm(enumerate(samples)):
        data.append(
            {
                "src": source_text,
                "mt": predictions[i].replace("```", ""),
                "ref": reference_text,
            }
        )
    model_output = model.predict(data, batch_size=2, gpus=1)
    return round(model_output.system_score * 100, 2)


def get_wmt23_cometkiwi_da_xl_score(samples: list[tuple[str, str]], predictions: list[str]):
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)
    data = []

    for i, (source_text, reference_text) in tqdm(enumerate(samples)):
        data.append(
            {
                "src": source_text,
                "mt": predictions[i].replace("```", ""),
            }
        )
    model_output = model.predict(data, batch_size=2, gpus=1)
    return round(model_output.system_score * 100, 2)


def get_bleurt_score(samples: list[tuple[str, str]], predictions: list[str], batch_size=100):
    model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20")
    tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20")

    references = [x[1] for x in samples]
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


def get_bert_score(samples: list[tuple[str, str]], predictions: list[str]):
    references = [x[1] for x in samples]
    precision, recall, f1 = score(predictions, references, model_type="bert-base-multilingual-cased", verbose=True)
    return round(f1.mean().item() * 100, 2)


def get_prediction(model: str, text: str, language_from: str, language_to: str, prompt_name: str = "Prompt 3"):
    translation_task = TranslationTask(text=text, language_from=language_from, language_to=language_to)
    content = get_content(translation_task, prompt_name)
    response = Client().chat(model=model, messages=[{"role": "user", "content": content}])
    return response["message"]["content"]


def benchmark(model: str, language_pair: str, limit: int = 2000, prompt_name: str = "Prompt 3"):
    predictions_path = Path(join(ROOT_PATH, "language_data", "predictions", model, f"samples_{limit}", language_pair))

    if not predictions_path.exists():
        os.makedirs(predictions_path)

    print(f"Model: {model}, Pair: {language_pair}")
    samples = read_samples(language_pair)
    if limit:
        samples = samples[:limit]
    translations = list()
    print(f"Number of samples: {len(samples)}")
    print("Starting benchmark...")
    total_time = 0
    for i, (from_text, human_translation) in tqdm(enumerate(samples)):
        batch = i // 50
        path = Path(join(predictions_path, str(batch) + ".json"))
        if path.exists():
            print("skipping batch", batch, "...")
            continue

        language_from = language_pair.split("-")[0]
        language_to = language_pair.split("-")[1]
        start_time = time()
        prediction = get_prediction(model, from_text, language_from, language_to, prompt_name)
        total_time += time() - start_time
        translations.append(prediction)

        if (i + 1) % 50 == 0:
            path.write_text(json.dumps(translations, indent=4))
            translations = list()

    print(f"Total time: {round(total_time, 2)} seconds")

    get_performance(samples, predictions_path, prompt_name, round(total_time, 2))


def get_performance(
    samples: list[tuple[str, str]], predictions_path: Path, prompt_name: str = "Prompt 3", total_time: float = 0.0
):
    results_path = Path(join(ROOT_PATH, "results", "model_benchmark_results.csv"))
    if not results_path.exists():
        results_path.write_text(
            "model,language_pair,sample_count,prompt_name,bleu,xcomet_xl,wmt23_cometkiwi_da_xl,bleurt,bert_score,average_score,total_time\n"
        )
    model = predictions_path.parent.parent.name
    sample_count = predictions_path.parent.name.split("_")[1]
    language_pair = predictions_path.name

    subprocess.run(["ollama", "stop", model])
    sleep(5)

    predictions = list()
    for file in sorted(os.listdir(predictions_path), key=lambda x: int(x.split(".")[0])):
        predictions += json.loads(Path(join(predictions_path, file)).read_text())

    print("Getting BLEU score...")
    bleu_score = get_bleu_score(samples, predictions)
    print("Getting XCOMET-XL score...")
    xcomet_xl_score = get_xcomet_xl_score(samples, predictions)
    print("Getting WMT23 COMETKIWI DA XL score...")
    wmt23_cometkiwi_da_xl_score = get_wmt23_cometkiwi_da_xl_score(samples, predictions)
    print("Getting BLEURT score...")
    bleurt_score = get_bleurt_score(samples, predictions)
    print("Getting BERT score...")
    bert_score = get_bert_score(samples, predictions)
    average_score = round((bleu_score + xcomet_xl_score + wmt23_cometkiwi_da_xl_score + bleurt_score + bert_score) / 5, 2)

    print(f"BLEU score: {bleu_score}")
    print(f"XCOMET-XL score: {xcomet_xl_score}")
    print(f"WMT23 COMETKIWI DA XL score: {wmt23_cometkiwi_da_xl_score}")
    print(f"BLEURT score: {bleurt_score}")
    print(f"BERT score: {bert_score}")

    result = f"{model},{language_pair},{sample_count},{prompt_name},{bleu_score},{xcomet_xl_score},{wmt23_cometkiwi_da_xl_score},{bleurt_score},{bert_score},{average_score},{total_time}\n"
    with open(results_path, "a") as f:
        f.write(result)

    torch.cuda.empty_cache()
    gc.collect()
    sleep(5)


if __name__ == "__main__":
    # start = time()
    # print("start")
    languages = ["en-es", "en-fr", "en-ru"]
    for pair in languages:
        benchmark("aya:35b", pair, 100)
    # print("time", round(time() - start, 2), "s")
    # print(read_samples("en-es", 10))

    # print(get_bleu_score("Can it be delivered between 10 to 15 minutes?", "Can I receive my food in 10 to 15 minutes?"))
