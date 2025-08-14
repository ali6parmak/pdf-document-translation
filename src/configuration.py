from pathlib import Path

SRC_PATH = Path(__file__).parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.absolute()

LANGUAGES_SHORT = ["en", "fr", "es", "ru", "ar", "sp"]
LANGUAGES = ["English", "French", "Spanish", "Russian", "Arabic", "Spanish"]


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
""",
}
