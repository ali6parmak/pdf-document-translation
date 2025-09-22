import json
from pathlib import Path
from benchmark_models import read_samples
from configuration import DATA_SOURCE_PATH, ROOT_PATH, LABELS_SOURCE_PATH
from comet import download_model, load_from_checkpoint


def load_data(data_path: Path | str):
    data = json.loads(Path(data_path).read_text())
    return data


def benchmark_human_performance():
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)

    results = []

    for data_path in sorted(LABELS_SOURCE_PATH.iterdir()):
        print(f"Processing {data_path.name}...")
        data = load_data(data_path)
        model_input = []
        for paragraph in data["paragraphs"]:
            model_input.append(
                {
                    "src": paragraph["main_language"],
                    "mt": paragraph["other_language"],
                }
            )

        model_output = model.predict(model_input, batch_size=2, gpus=1)
        score = round(model_output.system_score * 100, 2)

        results.append([data_path.name, data["main_language"], data["other_language"], score])

    result_path = Path(ROOT_PATH, "results", "human_performance_results.csv")
    results_string = "file_name,language_from,language_to,wmt23_cometkiwi_da_xl_score\n"
    results_string += "\n".join([",".join(map(str, result)) for result in results])
    results_string += "\n\n"

    average_score = round(sum([result[3] for result in results]) / len(results), 2)
    results_string += f"Average score: {average_score}\n"

    result_path.write_text(results_string)


def benchmark_human_performance_titles():
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)

    results = []

    for data_path in sorted(Path(DATA_SOURCE_PATH, "labels_titles").iterdir()):
        print(f"Processing {data_path.name}...")
        data = load_data(data_path)
        model_input = []
        for paragraph in data["paragraphs"]:
            model_input.append(
                {
                    "src": paragraph["main_language"],
                    "mt": paragraph["other_language"],
                }
            )
        model_output = model.predict(model_input, batch_size=2, gpus=1)
        score = round(model_output.system_score * 100, 2)
        results.append([data_path.name, data["main_language"], data["other_language"], score])

    result_path = Path(ROOT_PATH, "results", "human_performance_titles_results.csv")
    results_string = "file_name,language_from,language_to,wmt23_cometkiwi_da_xl_score\n"
    results_string += "\n".join([",".join(map(str, result)) for result in results])
    results_string += "\n\n"

    average_score = round(sum([result[3] for result in results]) / len(results), 2)
    results_string += f"Average score: {average_score}\n"

    result_path.write_text(results_string)


def benchmark_human_performance_language_data():
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)

    results = []
    language_pairs = ["ar-en", "en-es", "en-fr", "en-ru"]
    for language_pair in language_pairs:
        for data_path in sorted(Path(ROOT_PATH, "language_data", language_pair).iterdir()):
            print(f"Processing {data_path.name}...")
            samples = read_samples(language_pair)
            model_input = []
            for paragraph in samples:
                model_input.append(
                    {
                        "src": paragraph[0],
                        "mt": paragraph[1],
                    }
                )

            model_output = model.predict(model_input, batch_size=2, gpus=1)
            score = round(model_output.system_score * 100, 2)

            results.append([data_path.name, language_pair.split("-")[0], language_pair.split("-")[1], score])

    result_path = Path(ROOT_PATH, "results", "human_performance_language_data_results.csv")
    results_string = "file_name,language_from,language_to,wmt23_cometkiwi_da_xl_score\n"
    results_string += "\n".join([",".join(map(str, result)) for result in results])
    results_string += "\n\n"

    average_score = round(sum([result[3] for result in results]) / len(results), 2)
    results_string += f"Average score: {average_score}\n"

    result_path.write_text(results_string)


if __name__ == "__main__":
    benchmark_human_performance_language_data()
