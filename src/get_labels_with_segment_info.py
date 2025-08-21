import json
import subprocess
from pathlib import Path
from configuration import DATA_SOURCE_PATH, LABELS_SOURCE_PATH, PDFS_SOURCE_PATH


VGT_JSONS_PATH = Path(DATA_SOURCE_PATH, "vgt_jsons")


def get_vgt_segments():
    if not VGT_JSONS_PATH.exists():
        VGT_JSONS_PATH.mkdir(parents=True, exist_ok=True)

    for pdf_path in PDFS_SOURCE_PATH.iterdir():
        command = [
            "curl",
            "-X",
            "POST",
            "-F",
            f"file=@{pdf_path}",
            "localhost:5060",
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        json_data = json.loads(result.stdout)
        Path(VGT_JSONS_PATH, pdf_path.name.replace(".pdf", ".json")).write_text(json.dumps(json_data, indent=4))


def get_word_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    percentage = len(intersection) / len(set1) * 100
    return round(percentage, 2)


def assign_segments_to_labels():

    for label_path in LABELS_SOURCE_PATH.iterdir():
        label_data = json.loads(label_path.read_text())
        vgt_json_name = label_path.name.rsplit("_", 1)[0] + ".json"
        segment_data = json.loads(Path(VGT_JSONS_PATH, vgt_json_name).read_text())

        for paragraph in label_data["paragraphs"]:
            paragraph["main_language"] = " ".join(paragraph["main_language"].split())

        for segment in segment_data:
            segment["text"] = " ".join(segment["text"].split())

        for paragraph in label_data["paragraphs"]:
            best_matching_segment = None
            best_score = 0
            for segment in segment_data:
                if not segment["text"].strip():
                    continue

                similarity_score = get_word_similarity(paragraph["main_language"], segment["text"])

                if similarity_score > best_score:
                    best_score = similarity_score
                    best_matching_segment = segment

                if similarity_score > 90:
                    break

            paragraph["page_number"] = best_matching_segment["page_number"]
            paragraph["type"] = best_matching_segment["type"]
            if best_score == 0:
                print(f"Paragraph not found in {label_path.name}:\n{paragraph['main_language']}")
                print("*" * 50)

        Path(DATA_SOURCE_PATH, "labels_with_segment_info", label_path.name).write_text(json.dumps(label_data, indent=4))


if __name__ == "__main__":
    # get_vgt_segments()
    assign_segments_to_labels()
