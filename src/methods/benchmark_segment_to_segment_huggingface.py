import gc
import sys
import os

from benchmark_segment_to_segment import (
    get_bert_score,
    get_bleurt_score,
    get_wmt23_cometkiwi_da_xl_score,
    get_xcomet_xl_score,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
from pathlib import Path
from time import sleep, time
from configuration import LABELS_SOURCE_PATH, PREDICTIONS_SOURCE_PATH, LANGUAGES_SHORT, LANGUAGES, PROMPTS, ROOT_PATH
from domain.TranslationTask import TranslationTask
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from vllm import LLM, SamplingParams
from huggingface_hub import login
from dotenv import load_dotenv


METHOD_NAME = "segment_to_segment"


load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))


LANGUAGE_TO_FLORES = {
    "ar": "arb_Arab",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_nllb_prediction(
    model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, text: str, language_from: str, language_to: str
):

    tokenizer.src_lang = LANGUAGE_TO_FLORES[language_from]
    tgt_lang = LANGUAGE_TO_FLORES[language_to]
    inputs = tokenizer(text, return_tensors="pt").to(device)

    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang), max_length=512
    )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


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


def get_prediction(
    hf_model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, text: str, language_from: str, language_to: str
):
    return get_nllb_prediction(hf_model, tokenizer, text, language_from, language_to)


def get_predictions_for_file(data: dict, hf_model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer):
    for paragraph in data["paragraphs"]:
        prediction = get_prediction(
            hf_model, tokenizer, paragraph["main_language"], data["main_language"], data["other_language"]
        )
        paragraph["prediction"] = prediction
    return data


def get_bytedance_seed_x_ppo_prediction(hf_model: AutoModelForCausalLM, text: str, language_from: str, language_to: str):
    languages_from = [x for x in LANGUAGES_SHORT if language_from.lower()[:2] == x]
    source_language = LANGUAGES[LANGUAGES_SHORT.index(languages_from[0])]
    languages_to = [x for x in LANGUAGES_SHORT if language_to.lower()[:2] == x]
    target_language = LANGUAGES[LANGUAGES_SHORT.index(languages_to[0])]

    messages = [
        f"Translate the following {source_language} sentence into {target_language}:\n{text} <{language_to.lower()}>"
    ]

    decoding_params = SamplingParams(temperature=0, max_tokens=512, skip_special_tokens=True)
    results = hf_model.generate(messages, decoding_params)
    return results[0].outputs[0].text.strip()


def get_predictions_for_file_bytedance(data: dict, hf_model: LLM):

    for paragraph in data["paragraphs"]:
        prediction = get_bytedance_seed_x_ppo_prediction(
            hf_model, paragraph["main_language"], data["main_language"], data["other_language"]
        )
        paragraph["prediction"] = prediction
    return data


def get_model_predictions(model: str, prompt_name: str = "Prompt 3"):
    predictions_path = Path(PREDICTIONS_SOURCE_PATH, METHOD_NAME, model)
    if not predictions_path.exists():
        predictions_path.mkdir(parents=True, exist_ok=True)

    # model_name = "facebook/nllb-200-3.3B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # hf_model = hf_model.to(device)

    model_name = "ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8"
    hf_model = LLM(
        model=model_name, max_num_seqs=512, tensor_parallel_size=1, enable_prefix_caching=True, gpu_memory_utilization=0.95
    )

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
        # data = get_predictions_for_file(data, hf_model, tokenizer)
        data = get_predictions_for_file_bytedance(data, hf_model)
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
    model = "bytedance-seed-x-ppo-7b-gptq-int8"
    prompt_name = "ByteDanceSeedXPPOPrompt"

    get_model_predictions(model, prompt_name)
    # benchmark_model_translations(model, prompt_name)
