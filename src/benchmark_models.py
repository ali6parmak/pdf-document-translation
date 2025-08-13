import json
import os
from pathlib import Path
from time import time

import pandas as pd
from shutil import rmtree
from os.path import join

from tqdm import tqdm
from huggingface_hub import hf_hub_download

from configuration import ROOT_PATH, LANGUAGES_SHORT, LANGUAGES
from domain.TranslationTask import TranslationTask
from fast_bleu import BLEU
from ollama import Client

# MODELS = ["llama3", "tinyllama", "GLM-4"]
LANGUAGES_PAIRS = ["en-ru"]
# LANGUAGES_PAIRS = ["en-es"]

MODELS = ["aya:35b"]


def get_content(translation_task: TranslationTask):
    language_to_name = "English"
    languages_to = [x for x in LANGUAGES_SHORT if translation_task.language_to.lower()[:2] == x]

    if languages_to:
        language_to_name = LANGUAGES[LANGUAGES_SHORT.index(languages_to[0])]

    content = f"""Please translate the following text into {language_to_name}. Follow these guidelines:
1. Maintain the original layout and formatting.
2. Translate all text accurately without omitting any part of the content.
3. Preserve the tone and style of the original text.
4. Do not include any additional comments, notes, or explanations in the output; provide only the translated text.
5. Only translate the text between ``` and ```. Do not output any other text or character.

Here is the text to be translated:
"""
    content += "\n\n" + "```" + translation_task.text + "```"
    return content


def download_data():
    repo_id = "Helsinki-NLP/opus-100"
    train_file_name = "test-00000-of-00001.parquet"
    for pair in LANGUAGES_PAIRS:
        file_name = join(pair, train_file_name)
        hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=join(f"{ROOT_PATH}/data"), repo_type="dataset")
    rmtree(join(ROOT_PATH, "data", ".huggingface"))


def read_samples(language_pair: str, limit: int = 0) -> list[tuple[str, str]]:
    df = pd.read_parquet(join(ROOT_PATH, "data", language_pair), engine="pyarrow")
    lang1, lang2 = df.iloc[0]["translation"].keys()
    texts_translations = list()
    for i, row in df.iterrows():
        texts_translations.append((row["translation"][lang1], row["translation"][lang2]))
        if limit and i == limit:
            break

    return texts_translations


def get_bleu_score(correct_text: str, prediction: str):
    list_of_references = [correct_text.split()]
    hypotheses = [prediction.split()]
    weights = {"bigram": (1 / 2.0, 1 / 2.0), "trigram": (1 / 3.0, 1 / 3.0, 1 / 3.0)}
    bleu = BLEU(list_of_references, weights)
    average = (bleu.get_score(hypotheses)["bigram"][0] + bleu.get_score(hypotheses)["trigram"][0]) / 2.0
    return average


def get_prediction(model: str, text: str, language_from: str, language_to: str):
    translation_task = TranslationTask(text=text, language_from=language_from, language_to=language_to)
    content = get_content(translation_task)
    response = Client().chat(model=model, messages=[{"role": "user", "content": content}])
    return response["message"]["content"]


def benchmark(model: str, language_pair: str, limit: int = 0):
    root_path = Path(join(ROOT_PATH, "data", "predictions", model, language_pair))

    if not root_path.exists():
        os.makedirs(root_path)

    print(f"Model: {model}, Pair: {language_pair}")
    samples = read_samples(language_pair)
    if limit:
        samples = samples[:limit]
    translations = list()
    total_time = 0
    print(f"Number of samples: {len(samples)}")
    print("Starting benchmark...")
    for i, (from_text, human_translation) in tqdm(enumerate(samples)):
        batch = i // 50
        path = Path(join(root_path, str(batch) + ".json"))
        if path.exists():
            print("skipping batch", batch, "...")
            continue

        language_from = language_pair.split("-")[0]
        language_to = language_pair.split("-")[1]
        start_time = time()
        prediction = get_prediction(model, from_text, language_from, language_to)
        total_time += time() - start_time
        translations.append(prediction)

        if (i + 1) % 50 == 0:
            path.write_text(json.dumps(translations, indent=4))
            translations = list()

    print(f"Total time: {round(total_time, 2)} seconds")

    get_performance(samples, root_path)


def get_performance(samples: list[tuple[str, str]], path: Path):
    predictions = list()
    for file in sorted(os.listdir(path), key=lambda x: int(x.split(".")[0])):
        predictions += json.loads(Path(join(path, file)).read_text())
    average_bleu_performance = 0
    for i, (text_from, text_to) in tqdm(enumerate(samples)):
        prediction = predictions[i].replace("```", "")
        average_bleu_performance += get_bleu_score(text_to, prediction)

    print(f"Average bleuperformance: {100 * average_bleu_performance / len(samples)}")


if __name__ == "__main__":
    # start = time()
    # print("start")
    # benchmark("llama3.1:70b", "en-fr", 100)
    # print("time", round(time() - start, 2), "s")
    print(read_samples("en-es", 10))

    # print(get_bleu_score("Can it be delivered between 10 to 15 minutes?", "Can I receive my food in 10 to 15 minutes?"))
