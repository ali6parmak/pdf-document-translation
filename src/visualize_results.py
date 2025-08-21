import json
from pathlib import Path
from configuration import PREDICTIONS_SOURCE_PATH, VISUALIZATIONS_SOURCE_PATH


def make_table(colors: list[str], headers: list[str], paragraph: dict):
    paragraph["prediction"] = paragraph["prediction"].replace("`", "")
    cells = [
        f'<td style="background-color:{colors[0]}; color:black; border:2px solid black; vertical-align:top; width:33%;"><b>{headers[0]}</b><br>{paragraph["main_language"]}</td>',
        f'<td style="background-color:{colors[1]}; color:black; border:2px solid black; vertical-align:top; width:33%;"><b>{headers[1]}</b><br>{paragraph["other_language"]}</td>',
        f'<td style="background-color:{colors[2]}; color:black; border:2px solid black; vertical-align:top; width:33%;"><b>{headers[2]}</b><br>{paragraph["prediction"]}</td>',
    ]
    return "<table>\n<tr>\n" + "\n\n".join(cells) + "\n</tr>\n</table>\n"


def get_visualizations(prediction_path: Path):
    data = json.loads(prediction_path.read_text())

    colors = ["#e6f7ff", "#fffbe6", "#f6ffed"]  # Light blue  # Light yellow  # Light green
    model = prediction_path.parent.name
    headers = [
        f"Main Language [{data['main_language']}]",
        f"Other Language [{data['other_language']}]",
        f"Prediction [{model}]",
    ]

    markdown_content = ""
    for paragraph in data["paragraphs"]:
        markdown_content += make_table(colors, headers, paragraph)
        markdown_content += "\n\n"

    parent_path = VISUALIZATIONS_SOURCE_PATH / prediction_path.parent.parent.name / prediction_path.parent.name
    parent_path.mkdir(parents=True, exist_ok=True)
    Path(VISUALIZATIONS_SOURCE_PATH, parent_path, prediction_path.name.replace(".json", ".md")).write_text(markdown_content)


def visualize(predictions_path: Path):
    for predictions_path in predictions_path.iterdir():
        get_visualizations(predictions_path)


if __name__ == "__main__":
    visualize(Path(PREDICTIONS_SOURCE_PATH, "segment_to_segment", "gpt-oss"))
