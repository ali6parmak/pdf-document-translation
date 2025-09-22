<div style="font-size:18px;">

First of all, I have started with researching about the good metrics to be able to properly evaluate the methods/models we use. Up until this time, we were using `BLEU` to compare the results - which is a classic approach for this kind of task. Even though it's fast and a widely used approach, it mostly measures the n-gram overlap and not always well correlated with the human judgement, especially for high-quality or very different translations.


So I have done some research and found some models that are trained specifically for evaluating translation quality:

</div>



<details>
<summary><span style="font-size:20px"><strong>Models Used for Evaluation</strong></span></summary>

<br>

<div style="font-size:18px">

[XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL): Trained by `Unbabel`, this is an evaluation model that is trained to identify errors in sentences along with a final quality score and thus leading to an explainable neural metric. This is the XL version with `3.5B` parameters.

To be able to evaluate the results, this model requires:
- Original text
- Human translation (reference)
- Machine translation/prediction
- ~14 GB free GPU memory

</div>

---

<div style="font-size:18px">

[wmt23-cometkiwi-da-xl](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl): Trained by `Unbabel`, it receives a source sentence and the respective translation and returns a score that reflects the quality of the translation, so it's intented to be used for reference-free MT evaluation. It's a `3.5B` parameters model.

This model requires:

- Original text
- Machine translation/prediction
- ~14 GB free GPU memory 

</div>

---

<div style="font-size:18px">

[BLEURT](https://github.com/google-research/bleurt): Trained by `google-research`, the model takes a pair of sentences as input, a reference and a candidate, and it returns a score that indicates to what extent the candidate is fluent and conveys the meaning of the reference. It's a BERT-like [model](https://huggingface.co/lucadiliello/BLEURT-20) and is good for semantic similarity.

This model requires:

- Original text
- Human translation (reference)
- Machine translation/prediction
- ~2.5 GB free GPU memory 

</div>

---

<div style="font-size:18px">

[bert_score](https://github.com/Tiiiger/bert_score): Measures similarity at the token level using contextual embeddings. It's good for capturing meaning, but can sometimes be too forgiving. For this method, there is no single model, we can just use any BERT-like model and I have used `bert-base-multilingual-cased` model. You can see the models supported from [here](https://github.com/Tiiiger/bert_score/blob/master/bert_score/utils.py).

This model requires:

- Original text
- Human translation (reference)
- Machine translation/prediction
- A BERT-like model
  
</div>

</details>

---

<details>
<summary><span style="font-size:20px"><strong>Data</strong></span></summary>

<br>

<div style="font-size:18px">

To evaluate the models, we used 2 datasets. One is from [Helsinki-NLP](https://huggingface.co/datasets/Helsinki-NLP/opus-100). From this dataset, we have used:

- Arabic-to-English (2000 samples)
- English-to-Spanish (2000 samples)
- English-to-French (2000 samples)
- English-to-Russian (2000 samples)

_Every sample here is a `sentence`._

The other dataset is the one we created from our PDFs:

- cejil_1_es.pdf
- cejil_1_pt.pdf
- cejil_2_es.pdf
- cejil_2_pt.pdf
- ihrda_1_en.pdf
- ihrda_1_pt.pdf
- ihrda_2_en.pdf
- ihrda_2_fr.pdf
- ohchr_1_en.pdf
- ohchr_1_ru.pdf
- ohchr_2_en.pdf
- ohchr_2_fr.pdf
- plan_1_en.pdf
- plan_1_es.pdf
- plan_2_en.pdf
- plan_2_fr.pdf

What we did is, for the same document, we translated them in both ways. For example for cejil_1, we have "cejil_1_es_pt" and "cejil_1_pt_es".

_At total, we have 1691 samples here and every sample is a `paragraph`._


</div>

</details>

---


<details>
<summary><span style="font-size:20px"><strong>Evaluation</strong></span></summary>

<br>


<div style="font-size:18px">

Here are the results for Helsinki-NLP's dataset with _[ar-en, en-es, en-fr, en-ru]_ language pairs (2000 samples each):


| model                                    | sample_count | prompt_name             | xcomet_xl | wmt23_cometkiwi_da_xl | bleurt | bert_score | average_score | total_time  | device            |
|------------------------------------------|--------------|-------------------------|-----------|----------------------|--------|------------|---------------|-------------|--------------------|
| deepl                                    | 8000         | (no prompt)             | 90.04     | 73.18                | 69.66  | 86.48      | 79.84         | 3749.07     | -                  |
| bytedance-seed-x-ppo-7b-gptq-int8        | 8000         | ByteDanceSeedXPPOPrompt | 90.02     | 74.6                 | 68.66  | 85.72      | 79.75         | 4077.01     | RTX-4070 Ti SUPER  |
| nllb-200-3.3B                            | 8000         | (no prompt)             | 87.69     | 70.56                | 67.12  | 85.77      | 77.78         | 2912.45     | RTX-4070 Ti SUPER  |
| nllb-200-3.3B[cpu]                       | 8000         | (no prompt)             | 87.69     | 70.56                | 67.12  | 85.77      | 77.78         | 23887.72    | i7-14700K          |
| huihui_ai/hunyuan-mt-abliterated:7b      | 8000         | Prompt 3                | 88.9      | 74.59                | 65.51  | 82.03      | 77.76         | 3094.56     | RTX-4070 Ti SUPER  |
| aya:35b[gcloud]                          | 8000         | Prompt 3                | 88.24     | 72.63                | 67.46  | 82.16      | 77.62         | 96875.48    | NVIDIA - L4        |
| nllb-200-distilled-600M                  | 8000         | (no prompt)             | 86.47     | 69.46                | 65.6   | 85.36      | 76.72         | 5738.46     | RTX-4070 Ti SUPER  |
| gpt-oss                                  | 8000         | Prompt 3                | 86.56     | 72.34                | 65.96  | 78.02      | 75.72         | 37045.49    | RTX-4070 Ti SUPER  |
| zongwei/gemma3-translator:4b             | 8000         | Prompt 3                | 84.7      | 68.41                | 63.74  | 76.01      | 73.22         | 2382.86     | RTX-4070 Ti SUPER  |
| llama3.1                                 | 8000         | Prompt 3                | 82.04     | 66.29                | 61.9   | 75.11      | 71.34         | 2870.2      | RTX-4070 Ti SUPER  |
| trillionlabs:1.8b                        | 8000         | TrillionlabsPrompt      | 70.22     | 62.54                | 31.31  | 70.8       | 58.72         | 2507.71     | RTX-4070 Ti SUPER  |


DeepL is right now the most dependable translator existing out there so I used it's API and get the predictions to see which models are performing closer to that. Of course the evaluation models that we use have their own flaws so the numbers we see under the "average_score" column does not actually mean the "actual score of a model", instead, we use this as some score to compare the models between each other. And instead of using a single model to evaluate, we average the scores coming from different models and by doing this we try to achieve more consistent - correct results.

As you can see, according to our evaluation models, [bytedance-seed-x-ppo-7b-gptq-int8](https://huggingface.co/ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8) performing very close to what DeepL performs. And it even performs better than that according to `wmt23_cometkiwi_da_xl`. This model is a quantized version of [this](https://huggingface.co/ByteDance-Seed/Seed-X-PPO-7B) model. I used the quantized version because it was not fitting my GPU, so the actual performance of this model can be slightly better.


And here are the results for our dataset:

| model                                      | method                        | prompt_name                | total_average_score | total_time_passed | total_sample_count | time_per_sample | device              |
|---------------------------------------------|-------------------------------|----------------------------|--------------------|-------------------|-------------------|-----------------|---------------------|
| bytedance-seed-x-ppo-7b-gptq-int8           | segment_to_segment            | ByteDanceSeedXPPOPrompt    | 82.57              | 3276.22           | 1691              | 1.94            | RTX-4070 Ti SUPER   |
| aya:35b[gcloud]                             | segment_to_segment            | Prompt 3                   | 80.33              | 67529.32          | 1691              | 39.93           | NVIDIA - L4         |
| huihui_ai/hunyuan-mt-abliterated:7b         | segment_to_segment            | Prompt 3                   | 80.33              | 2209.85           | 1691              | 1.31            | RTX-4070 Ti SUPER   |
| gpt-oss                                     | page_to_page                  | Prompt 3                   | 80.08              | 8221.43           | 265               | 31.02           | RTX-4070 Ti SUPER   |
| gpt-oss                                     | segment_to_segment            | Prompt 3                   | 79.93              | 13244.2           | 1691              | 7.83            | RTX-4070 Ti SUPER   |
| gpt-oss                                     | previous_after_segments       | Prompt 4                   | 79.84              | 13320.63          | 1691              | 7.88            | RTX-4070 Ti SUPER   |
| gpt-oss                                     | segment_to_segment_with_titles| Prompt 5                   | 79.08              | 12930.97          | 1691              | 7.65            | RTX-4070 Ti SUPER   |
| nllb-200-3.3B                               | segment_to_segment            | (no prompt)                | 79.03              | 2358.87           | 1691              | 1.39            | RTX-4070 Ti SUPER   |
| zongwei/gemma3-translator:4b                | segment_to_segment            | Prompt 3                   | 74.54              | 1298.0            | 1691              | 0.77            | RTX-4070 Ti SUPER   |


Up until now, we have used `aya:35b` as our translator but we can see that there is a 2% difference with `bytedance-seed-x-ppo-7b-gptq-int8`, which I think is a big difference.

</div>


</details>


---


<details>
<summary><span style="font-size:20px"><strong>Notes About Models</strong></span></summary>

<br>


<div style="font-size:18px">


- `aya:35b` is a good model which trained on 23 languages and currently our translation instances still use it. But what I don't like about it is that compared to especially `gpt-oss`, `bytedance-seed-x-ppo-7b-gptq-int8` and `huihui_ai/hunyuan-mt-abliterated:7b` it's very slow. To actually see it's speed, I also run the `gpt-oss` model in the same environment with `aya` and gpt-oss was `~3.55 times` faster.

- `gpt-oss` is a decent model, though it's not performing as well as the others. Maybe not it's translation capabilities but I think this model might be better in following instructions and understanding what user wants. So it's more like a general-purpose model.

- `bytedance-seed-x-ppo-7b-gptq-int8` is a very good model. And according to the model's documentation:

    >This model is specialized in multilingual translation, which is unexpected to support other tasks.

    Which makes it great for translation task. And even though this is a quantized version of the original model, it still performs much better than the other models, closer to what we get from DeepL. Also, based on the results I shared above, it works `~9 times` faster than `gpt-oss` in my local computer (with RTX 4070 Ti SUPER). Which makes it (theoratically) `32 times` faster than `aya:35b`.

- I also really liked the model `huihui_ai/hunyuan-mt-abliterated:7b`. The original model is [here](https://huggingface.co/tencent/Hunyuan-MT-7B), which is like a ~15 GB model but it was not fitting to my GPU. So someone shared the "abliterated" version of it in ollama, [here](https://ollama.com/huihui_ai/hunyuan-mt-abliterated), and it's only ~4.6 GB. I think it's a pretty decent model, `25%` faster than `bytedance-seed-x-ppo-7b-gptq-int8` and still performs better than `aya:35b`. And since it has a very small size, we can use it as a default model in `pdf-document-layout-analysis` to translate documents to another language. So that users do not have to wait some ~20 GB model to download.
- `nllb-200-3.3B` also is an interesting model. It has been trained by `facebook` and they claim that the model is supporting 200 languages. As for the results, it's acting slightly better in the Helsinki-NLP's dataset but a bit worse than `aya:35b` in our dataset but still, if we need a model for some uncommon, underrepresented language, this can be a choice.
- I tried different methods like `segment-to-segment`, `previous-after-segments`, `page-to-page`, `segment-to-segment-with-titles`. I tried these methods only with `gpt-oss` but since there was not a big difference between them, I did not try these methods with the other models.


</div>


</details>


---

<details>
<summary><span style="font-size:20px"><strong>Prompts</strong></span></summary>

<br>


<div style="font-size:18px">

`Prompt 1:`

```
 Translate the below text to {language_to_name}, keep the layout, do not skip any text, do not output anything else besides translation:
```    

`Prompt 2:`
```
Please translate the following text into {language_to_name}. Follow these guidelines:  
      1. Maintain the original layout and formatting.  
      2. Translate all text accurately without omitting any part of the content.  
      3. Preserve the tone and style of the original text.  
      4. Do not include any additional comments, notes, or explanations in the output; provide only the translated text.  

Here is the text to be translated: 
```

`Prompt 3:`

```
Please translate the following text into {language_to_name}. Follow these guidelines:
1. Maintain the original layout and formatting.
2. Translate all text accurately without omitting any part of the content.
3. Preserve the tone and style of the original text.
4. Do not include any additional comments, notes, or explanations in the output; provide only the translated text.
5. Only translate the text between ``` and ```. Do not output any other text or character.

Here is the text to be translated:

\```
{text_to_translate}
\```
```

`Prompt 4:`

```
Please translate only the text marked as "TARGET SEGMENT" into {language_to_name}. Use the "PREVIOUS SEGMENT" and "NEXT SEGMENT" only as context to help you understand the meaning, but do not translate them. 

Guidelines:
1. Maintain the original layout and formatting of the TARGET SEGMENT.
2. Translate all text in the TARGET SEGMENT accurately without omitting any part of the content.
3. Preserve the tone and style of the TARGET SEGMENT.
4. Do not include any additional comments, notes, or explanations in the output; provide only the translated TARGET SEGMENT.
5. The "PREVIOUS SEGMENT" and "NEXT SEGMENT" are provided only for context and may be `[empty]`.

Context:
PREVIOUS SEGMENT:
\```
{previous_text}
\```

TARGET SEGMENT (translate only this part):
\```
{text_to_translate}
\```

NEXT SEGMENT:
\```
{next_text}
\```
```

`Prompt 5:`

```

Please translate the following text into {language_to_name}. The text is an excerpt from a document with the following title:

{document_title}


Follow these guidelines:

1. Maintain the original layout and formatting.
2. Translate all text accurately without omitting any part of the content.
3. Preserve the tone and style of the original text.
4. Do not include any additional comments, notes, or explanations in the output; provide only the translated text.
5. Only translate the text between ``` and ```. Do not output any other text or character.

Here is the text to be translated:

\```
{text_to_translate}
\```

```

Some models have their own formats instead of custom prompts:

`ByteDanceSeedXPPOPrompt`:
```
f"Translate the following {source_language} sentence into {target_language}:\n{text} <{target_language_code}>"
```

`TrillionLabsPrompt`:

```
f"Translate the following {source_language} text into {target_language}:\n{text} <{target_language_code}>"
```

And instead of using a prompt, getting the results with `nllb-200-3.3B` is like:

```
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
```
</div>


</details>


---

<details>
<summary><span style="font-size:20px"><strong>Conclusion</strong></span></summary>

<br>


<div style="font-size:18px">


After testing different kinds of translation models and evaluation methods, it's clear that we should not only rely on BLEU while evaluating the models, it misses important details and the actual meaning in the translation. We get a much better sense of real translation quality with these newer methods/models.

From all the models I have tested, `bytedance-seed-x-ppo-7b-gptq-int8` is the one stood out most. It's fast, accurate, and gets results very close to DeepL, which is kind of known as the best translator right now. 

`hunyuan-mt-abliterated:7b` is also promising, especially if we need something lightweight. It's only 4.6 GB and performs slightly better than the current model we use, `aya:35b`.

If we want some general purpose model, `gpt-oss` can be a way to go. It also has decent scores on translation task but there are better models.

If we need to translate some underrepresented languages, we can go with `nllb-200-3.3B`. Their author, `facebook` claimed that it supports 200 languages and it has decent scores on the data we have.

Trying different methods like `page-to-page` or `previous-after-segments` did not affect the results much. So we can maybe continue with just `segment-to-segment` approach. Since it's easier to control and it's easier for models to translate little parts, without losing anything in the translation. But of course, we can try to think new ways about it and also if we decide to switch to a new model, I can try these different methods on that model to see how does it perform.


Overall, switching to `bytedance-seed-x-ppo-7b-gptq-int8` for most of our translation tasks makes sense. Of course we should keep an eye on new models but for now, this looks like the way to go.

</div>

</details>