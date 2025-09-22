import pandas as pd
from pathlib import Path
from configuration import ROOT_PATH


def get_translation_result_summary():

    df = pd.read_csv(Path(ROOT_PATH, "results", "model_translation_benchmarks.csv"))

    result = (
        df.groupby(["model", "method", "prompt_name"])["average_score"]
        .mean()
        .reset_index()
        .rename(columns={"average_score": "total_average_score"})
    )

    result["total_average_score"] = result["total_average_score"].round(2)

    result.to_csv(Path(ROOT_PATH, "results", "model_translation_benchmarks_summary.csv"), index=False)

    df_time = pd.read_csv(Path(ROOT_PATH, "results", "model_translation_times.csv"))

    agg_time = (
        df_time.groupby(["model", "method", "prompt_name"])
        .agg(total_time_passed=("time", "sum"), total_sample_count=("sample_count", "sum"))
        .reset_index()
    )

    agg_time["time_per_sample"] = agg_time["total_time_passed"] / agg_time["total_sample_count"]

    summary_df = pd.merge(result, agg_time, on=["model", "method", "prompt_name"])

    summary_df["time_per_sample"] = summary_df["time_per_sample"].round(2)
    summary_df["total_time_passed"] = summary_df["total_time_passed"].round(2)
    summary_df.sort_values(by="total_average_score", ascending=False, inplace=True)

    print(summary_df)

    summary_df.to_csv(Path(ROOT_PATH, "results", "model_translation_benchmarks_summary.csv"), index=False)


def get_benchmark_result_summary():
    df = pd.read_csv(Path(ROOT_PATH, "results", "model_benchmark_results.csv"))

    df = df.drop(columns=["bleu"])

    metric_cols = ["xcomet_xl", "wmt23_cometkiwi_da_xl", "bleurt", "bert_score"]

    grouped = (
        df.groupby(["model", "sample_count", "prompt_name"])
        .agg({**{col: "mean" for col in metric_cols}, "total_time": "sum"})
        .reset_index()
    )

    grouped["average_score"] = grouped[metric_cols].mean(axis=1)

    for col in metric_cols + ["average_score"]:
        grouped[col] = grouped[col].round(2)
    grouped["total_time"] = grouped["total_time"].round(2)

    # Rename the column here
    grouped = grouped.rename(columns={"sample_count": "sample_count_per_language_pair"})

    cols = ["model", "sample_count_per_language_pair", "prompt_name"] + metric_cols + ["average_score", "total_time"]
    grouped = grouped[cols]
    grouped.sort_values(by=["sample_count_per_language_pair", "average_score"], ascending=False, inplace=True)

    grouped.to_csv(Path(ROOT_PATH, "results", "model_benchmark_results_summary.csv"), index=False)
    print(grouped)


if __name__ == "__main__":
    get_translation_result_summary()
    get_benchmark_result_summary()
