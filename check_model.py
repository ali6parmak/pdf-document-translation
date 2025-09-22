import os
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


def check_bytedance_seed_x_ppo():
    model_path = "ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8"
    model = LLM(
        model=model_path, max_num_seqs=512, tensor_parallel_size=1, enable_prefix_caching=True, gpu_memory_utilization=0.95
    )
    content = """The goal of this task is to recognize human actions such as “ice skating” and “playing guitar” in video recordings. We use the most recent version of the Kinetics dataset (Kinetics700), a widely used benchmark for action classification [22]. It is a large-scale dataset with over 550k 10-second clips from 700 action classes. This problem has been primarily tackled from a visual perspective, although some multimodal approaches have also been proposed [23, 24]."""
    messages = [f"Translate the following English sentence into Turkish:\n{content} <tr>"]
    # Beam search (We recommend using beam search decoding)
    # decoding_params = BeamSearchParams(beam_width=4,
    #                                 max_tokens=512)
    # Greedy decoding
    decoding_params = SamplingParams(temperature=0, max_tokens=512, skip_special_tokens=True)
    results = model.generate(messages, decoding_params)
    # responses = [res.outputs[0].text.strip() for res in results]
    print(results[0].outputs[0].text.strip())


def check_trillionlabs_1_8b():
    model_name = "trillionlabs/Tri-1.8B-Translation"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    text = "안녕하세요"
    messages = [{"role": "user", "content": f"Translate the following Korean text into English:\n{text} <en>"}]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)

    outputs = model.generate(inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)

    print(f"Korean: {text}")
    print(f"English: {translation}")


def check_nllb_distilled_600m():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Set the source language
    tokenizer.src_lang = "eng_Latn"  # English
    tgt_lang = "fra_Latn"  # French

    text = "UN Chief says there is no military solution in Syria"

    # Tokenize input (no src_lang argument here)
    inputs = tokenizer(text, return_tensors="pt")

    # Generate translation
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang), max_length=512
    )

    # Decode and print
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(translated_text)


def check_hunyuan_mt_7b():
    model_name_or_path = "tencent/Hunyuan-MT-7B-fp8"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")  # You may want to use bfloat16 and/or move to GPU here
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(
        device
    )  # You may want to use bfloat16 and/or move to GPU here
    messages = [
        {
            "role": "user",
            "content": "Translate the following segment into Turkish, without additional explanation.\n\nIt’s on the house.",
        },
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")

    # outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)
    outputs = model.generate(tokenized_chat.to(device), max_new_tokens=2048)
    output_text = tokenizer.decode(outputs[0])

    answer_start_index = output_text.index("<|extra_0|>")
    answer_end_index = output_text.index("<|eos|>")
    answer = output_text[answer_start_index + 11 : answer_end_index]

    print(answer)
    print(model.hf_device_map)


def check_hunyuan_mt_7b_fp8():
    # If you want to load fp8 model with transformers, you need to change the name"ignored_layers" in config.json to "ignore" and upgrade the compressed-tensors to compressed-tensors-0.11.0.
    # ~/.cache/huggingface/hub/models--tencent--Hunyuan-MT-7B-fp8/snapshots/81e5a3f7199524570ba75e61360e990ba88665e4/config.json
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model_name_or_path = "tencent/Hunyuan-MT-7B-fp8"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto"
    )  # You may want to use bfloat16 and/or move to GPU here
    messages = [
        {
            "role": "user",
            "content": "Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house.",
        },
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")

    outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)
    output_text = tokenizer.decode(outputs[0])

    print(output_text)
    print(model.hf_device_map)


if __name__ == "__main__":
    check_bytedance_seed_x_ppo()
    # check_trillionlabs_1_8b()
    # check_nllb_distilled_600m()
    # check_hunyuan_mt_7b()
    # check_hunyuan_mt_7b_fp8()
