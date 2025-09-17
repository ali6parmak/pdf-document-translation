import gc
import json
import os
import torch
from vllm import LLM, SamplingParams
from benchmark_models import (
    get_bert_score,
    get_bleu_score,
    get_bleurt_score,
    get_wmt23_cometkiwi_da_xl_score,
    get_xcomet_xl_score,
    read_samples,
)
from pathlib import Path
from time import time, sleep
from os.path import join
from tqdm import tqdm
from configuration import ROOT_PATH, LANGUAGES_SHORT, LANGUAGES
from domain.TranslationTask import TranslationTask
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


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


TRILLIONLABS_CONTENT = (
    """Translate the following {source_language} text into {target_language}:\n{text} <{target_language_code}>"""
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_content(translation_task: TranslationTask):
    languages_from = [x for x in LANGUAGES_SHORT if translation_task.language_from.lower()[:2] == x]
    source_language = LANGUAGES[LANGUAGES_SHORT.index(languages_from[0])]
    languages_to = [x for x in LANGUAGES_SHORT if translation_task.language_to.lower()[:2] == x]
    target_language = LANGUAGES[LANGUAGES_SHORT.index(languages_to[0])]

    content = TRILLIONLABS_CONTENT.format(
        source_language=source_language,
        target_language=target_language,
        target_language_code=translation_task.language_to.lower(),
        text=translation_task.text,
    )
    return content


def get_trillionlabs_prediction(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str, content: str):
    messages = [{"role": "user", "content": content}]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)

    outputs = model.generate(inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    translation = tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)

    return translation


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


def get_hunyuan_mt_7b_prediction(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, translation_task: TranslationTask):
    languages_to = [x for x in LANGUAGES_SHORT if translation_task.language_to.lower()[:2] == x]
    target_language = LANGUAGES[LANGUAGES_SHORT.index(languages_to[0])]

    content = (
        f"Translate the following segment into {target_language}, without additional explanation.\n\n{translation_task.text}"
    )

    messages = [{"role": "user", "content": content}]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=2048)
    output_text = tokenizer.decode(outputs[0])
    answer_start_index = output_text.index("<|extra_0|>")
    answer_end_index = output_text.index("<|eos|>")
    answer = output_text[answer_start_index + 11 : answer_end_index]
    return answer


def get_prediction(
    hf_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str, language_from: str, language_to: str
):
    # translation_task = TranslationTask(text=text, language_from=language_from, language_to=language_to)
    # content = get_content(translation_task)
    # return get_trillionlabs_prediction(hf_model, tokenizer, text, content)
    return get_nllb_prediction(hf_model, tokenizer, text, language_from, language_to)
    # return get_hunyuan_mt_7b_prediction(hf_model, tokenizer, translation_task)


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


def benchmark(model: str, language_pair: str, limit: int = 2000, prompt_name: str = "Prompt 3"):

    # model_name = "trillionlabs/Tri-1.8B-Translation"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # model_name = "facebook/nllb-200-3.3B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # hf_model = hf_model.to(device)

    # model_name = "tencent/Hunyuan-MT-7B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    model_name = "ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8"
    hf_model = LLM(
        model=model_name, max_num_seqs=512, tensor_parallel_size=1, enable_prefix_caching=True, gpu_memory_utilization=0.95
    )

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
        # prediction = get_prediction(hf_model, tokenizer, from_text, language_from, language_to)
        prediction = get_bytedance_seed_x_ppo_prediction(hf_model, from_text, language_from, language_to)
        total_time += time() - start_time
        translations.append(prediction)

        if (i + 1) % 50 == 0:
            path.write_text(json.dumps(translations, indent=4))
            translations = list()

    print(f"Total time: {round(total_time, 2)} seconds")

    # get_performance(samples, predictions_path, prompt_name, round(total_time, 2))
    return round(total_time, 2)


def get_performance(
    model: str, language_pair: str, limit: int = 2000, prompt_name: str = "Prompt 3", total_time: float = 0.0
):

    samples = read_samples(language_pair)
    if limit:
        samples = samples[:limit]
    predictions_path = Path(join(ROOT_PATH, "language_data", "predictions", model, f"samples_{limit}", language_pair))

    results_path = Path(join(ROOT_PATH, "results", "model_benchmark_results.csv"))
    if not results_path.exists():
        results_path.write_text(
            "model,language_pair,sample_count,prompt_name,bleu,xcomet_xl,wmt23_cometkiwi_da_xl,bleurt,bert_score,average_score,total_time\n"
        )
    model = predictions_path.parent.parent.name
    sample_count = predictions_path.parent.name.split("_")[1]
    language_pair = predictions_path.name

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
    language_pairs = ["en-fr", "en-ru"]
    model_name = "bytedance-seed-x-ppo-7b-gptq-int8"
    # language_pair = "ar-en"
    limit = 2000
    prompt_name = "ByteDanceSeedXPPOPrompt"
    for language_pair in language_pairs:
        total_time = benchmark(model_name, language_pair, limit, prompt_name)
        get_performance(model_name, language_pair, limit, prompt_name, total_time)

    # print(read_samples("en-es", 10))

    # print(get_bleu_score("Can it be delivered between 10 to 15 minutes?", "Can I receive my food in 10 to 15 minutes?"))
