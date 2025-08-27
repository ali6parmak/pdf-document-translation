import json
from pathlib import Path
from configuration import DATA_SOURCE_PATH


LABELS_TITLES_PATH = Path(DATA_SOURCE_PATH, "labels_titles")
VGT_JSONS_PATH = Path(DATA_SOURCE_PATH, "vgt_jsons")


def get_empty_title_segment() -> dict:
    return {
        "left": 0,
        "top": 0,
        "width": 0,
        "height": 0,
        "page_number": 0,
        "page_width": 0,
        "page_height": 0,
        "text": "Empty title segment",
        "type": "Title",
    }


def create_title_labels():
    file_names = [f.name for f in sorted(VGT_JSONS_PATH.iterdir())]
    file_base_names = [f.rsplit("_", 1)[0] for f in file_names]
    for file_base_name in file_base_names:
        files_with_base_name = [f for f in file_names if file_base_name in f]
        if len(files_with_base_name) != 2:
            print(f"Skipping {file_base_name} because it has {len(files_with_base_name)} files")
            continue
        first_file_name = files_with_base_name[0]
        second_file_name = files_with_base_name[1]
        language_1 = first_file_name.rsplit("_", 1)[-1].split(".")[0]
        language_2 = second_file_name.rsplit("_", 1)[-1].split(".")[0]
        first_file_data = json.loads(Path(VGT_JSONS_PATH, first_file_name).read_text())
        second_file_data = json.loads(Path(VGT_JSONS_PATH, second_file_name).read_text())

        first_file_title_segments = [
            segment for segment in first_file_data if segment["type"] in ["Title", "Section header"]
        ]
        second_file_title_segments = [
            segment for segment in second_file_data if segment["type"] in ["Title", "Section header"]
        ]

        max_length = max(len(first_file_title_segments), len(second_file_title_segments))

        for i in range(max_length - len(first_file_title_segments)):
            first_file_title_segments.append(get_empty_title_segment())
        for i in range(max_length - len(second_file_title_segments)):
            second_file_title_segments.append(get_empty_title_segment())

        data = {
            "main_language": language_1,
            "other_language": language_2,
            "main_xml_name": first_file_name.replace(".json", ".xml"),
            "other_xml_name": second_file_name.replace(".json", ".xml"),
            "paragraphs": [],
        }

        for i in range(max_length):
            first_file_title_segment = first_file_title_segments[i]
            second_file_title_segment = second_file_title_segments[i]
            data["paragraphs"].append(
                {
                    "main_language": first_file_title_segment["text"],
                    "other_language": second_file_title_segment["text"],
                }
            )
        Path(LABELS_TITLES_PATH, f"{file_base_name}_{language_1}_{language_2}.json").write_text(json.dumps(data, indent=4))

        data_2 = {
            "main_language": language_2,
            "other_language": language_1,
            "main_xml_name": second_file_name.replace(".json", ".xml"),
            "other_xml_name": first_file_name.replace(".json", ".xml"),
            "paragraphs": [],
        }
        for i in range(max_length):
            second_file_title_segment = second_file_title_segments[i]
            first_file_title_segment = first_file_title_segments[i]
            data_2["paragraphs"].append(
                {
                    "main_language": second_file_title_segment["text"],
                    "other_language": first_file_title_segment["text"],
                }
            )
        Path(LABELS_TITLES_PATH, f"{file_base_name}_{language_2}_{language_1}.json").write_text(json.dumps(data_2, indent=4))


def reverse_labels(label_path: Path):
    label_data = json.loads(label_path.read_text())
    base_name = label_path.name.rsplit("_", 2)[0]
    new_label_path = Path(
        LABELS_TITLES_PATH, f"{base_name}_{label_data['other_language']}_{label_data['main_language']}.json"
    )
    label_data["main_language"], label_data["other_language"] = label_data["other_language"], label_data["main_language"]
    label_data["main_xml_name"], label_data["other_xml_name"] = label_data["other_xml_name"], label_data["main_xml_name"]

    for paragraph in label_data["paragraphs"]:
        paragraph["main_language"], paragraph["other_language"] = paragraph["other_language"], paragraph["main_language"]

    new_label_path.write_text(json.dumps(label_data, indent=4))


if __name__ == "__main__":
    # create_title_labels()
    reverse_labels(Path(LABELS_TITLES_PATH, "plan_2_en_fr.json"))
