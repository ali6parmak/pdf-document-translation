import pandas as pd
from pathlib import Path
from configuration import ROOT_PATH


def get_result_summary():

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


if __name__ == "__main__":
    get_result_summary()
