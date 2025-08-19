from pathlib import Path

SRC_PATH = Path(__file__).parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.absolute()

LANGUAGES_SHORT = ["en", "fr", "es", "ru", "ar", "sp", "pt"]
LANGUAGES = ["English", "French", "Spanish", "Russian", "Arabic", "Spanish", "Portuguese"]


DATA_SOURCE_PATH = Path(ROOT_PATH, "data")
PDFS_SOURCE_PATH = Path(DATA_SOURCE_PATH, "pdfs")
LABELS_SOURCE_PATH = Path(DATA_SOURCE_PATH, "labels")
PREDICTIONS_SOURCE_PATH = Path(DATA_SOURCE_PATH, "predictions")


PROMPTS = {
    "Prompt 1": "Translate the below text to {language_to_name}, keep the layout, do not skip any text, do not output anything else besides translation:",
    "Prompt 2": """Please translate the following text into {language_to_name}. Follow these guidelines:  
      1. Maintain the original layout and formatting.  
      2. Translate all text accurately without omitting any part of the content.  
      3. Preserve the tone and style of the original text.  
      4. Do not include any additional comments, notes, or explanations in the output; provide only the translated text.  

Here is the text to be translated:  """,
    "Prompt 3": """Please translate the following text into {language_to_name}. Follow these guidelines:
1. Maintain the original layout and formatting.
2. Translate all text accurately without omitting any part of the content.
3. Preserve the tone and style of the original text.
4. Do not include any additional comments, notes, or explanations in the output; provide only the translated text.
5. Only translate the text between ``` and ```. Do not output any other text or character.

Here is the text to be translated:

```
{text_to_translate}
```
""",
    "Prompt 4": """Please translate only the text marked as "TARGET SEGMENT" into {language_to_name}. Use the "PREVIOUS SEGMENT" and "NEXT SEGMENT" only as context to help you understand the meaning, but do not translate them. 

Guidelines:
1. Maintain the original layout and formatting of the TARGET SEGMENT.
2. Translate all text in the TARGET SEGMENT accurately without omitting any part of the content.
3. Preserve the tone and style of the TARGET SEGMENT.
4. Do not include any additional comments, notes, or explanations in the output; provide only the translated TARGET SEGMENT.
5. The "PREVIOUS SEGMENT" and "NEXT SEGMENT" are provided only for context and may be `[empty]`.

Context:
PREVIOUS SEGMENT:
```
{previous_text}
```

TARGET SEGMENT (translate only this part):
```
{text_to_translate}
```

NEXT SEGMENT:
```
{next_text}
```
""",
}
