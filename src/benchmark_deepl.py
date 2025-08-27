import gc
from time import sleep, time
import deepl
import os
import json
from pathlib import Path
from dotenv import load_dotenv

import torch
from tqdm import tqdm

from benchmark_models import (
    get_bert_score,
    get_bleu_score,
    get_bleurt_score,
    get_wmt23_cometkiwi_da_xl_score,
    get_xcomet_xl_score,
    read_samples,
)
from configuration import ROOT_PATH

load_dotenv()


def get_deepl_predictions(language_pair: str, sample_count: int = 2000):
    samples = read_samples(language_pair)
    translations = list()
    results_path = Path(ROOT_PATH, "language_data", "predictions", "deepl", f"samples_{sample_count}", language_pair)
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    for i, (text_from, _) in tqdm(enumerate(samples)):
        batch = i // 50
        print("batch", batch, "...")
        path = Path(results_path, str(batch) + ".json")

        if path.exists():
            print("skipping batch", batch, "...")
            continue

        translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))

        source_lang = language_pair.split("-")[0]

        target_lang = language_pair.split("-")[1]
        target_lang = target_lang if target_lang != "en" else "en-US"

        result = translator.translate_text(text_from, source_lang=source_lang, target_lang=target_lang)
        translations.append(result.text)

        if (i + 1) % 50 == 0:
            path.write_text(json.dumps(translations, indent=4))
            translations = list()


def get_performance(language_pair: str, sample_count: int, total_time: float = 0.0):

    samples = read_samples(language_pair)

    predictions = list()
    path = Path(ROOT_PATH, "deepl_results", language_pair)
    for file in sorted(os.listdir(path), key=lambda x: int(x.split(".")[0])):
        predictions += json.loads(Path(path, file).read_text())

    print(len(predictions))

    results_path = Path(ROOT_PATH, "results", "model_benchmark_results.csv")

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

    result = f"deepl,{language_pair},{sample_count},(no prompt),{bleu_score},{xcomet_xl_score},{wmt23_cometkiwi_da_xl_score},{bleurt_score},{bert_score},{average_score},{total_time}\n"
    with open(results_path, "a") as f:
        f.write(result)

    torch.cuda.empty_cache()
    gc.collect()
    sleep(5)


if __name__ == "__main__":
    start_time = time()
    get_deepl_predictions("en-fr")
    print(f"Time taken: {round(time() - start_time, 2)} seconds")
