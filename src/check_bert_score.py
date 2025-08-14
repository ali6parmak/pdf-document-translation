from bert_score import score


def check_translation():
    references = ["Can it be delivered between 10 to 15 minutes?", "Soon the sun will set"]
    candidates = ["Can you send it for 10 to 15 minutes?", "Merhaba"]
    precision, recall, f1 = score(candidates, references, model_type="bert-base-multilingual-cased", verbose=True)
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    print(f"System level F1 score: {round(f1.mean().item() * 100, 2)}")


if __name__ == "__main__":
    check_translation()
    # https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages
    # https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb
