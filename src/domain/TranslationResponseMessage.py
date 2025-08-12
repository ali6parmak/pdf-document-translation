from data_model.Translation import Translation
from data_model.TranslationTaskMessage import TranslationTaskMessage


class TranslationResponseMessage(TranslationTaskMessage):
    translations: list[Translation]
