# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: py313
#     language: python
#     name: py313
# ---

# %%
import re

import pandas as pd


def parse_log_to_dataframe(log_content: str) -> pd.DataFrame:
    """
    Parses a log string with multiple blocks of performance data into a pandas DataFrame.

    Each block is expected to start with a header line indicating the test name
    and parameters (M, K, NUM_THREADS), followed by a timing summary line,
    and then several key-value pairs for detailed metrics.
    Handles time units s, ms, and µs for timing data, storing them as Timedelta objects.
    Ignores lines that do not match expected patterns (e.g., informational messages).
    Correctly parses "loop" or "loops" in the timing line.

    Args:
        log_content: A string containing the entire log data.

    Returns:
        A pandas DataFrame where each row corresponds to a block in the log.
        The 's' and 'ds' columns will contain pandas.Timedelta objects.
    """
    records = []
    current_record = {}

    lines = log_content.strip().split("\n")

    header_pattern = re.compile(r"=========== (.*?) M=(\d+), K=(\d+), NUM_THREADS=(\d+) ===========")
    # Updated timing_pattern to handle "loop" or "loops"
    timing_pattern = re.compile(
        r"([\d.]+) (µs|ms|s) ± ([\d.]+) (µs|ms|s) per loop "
        r"mean±std\.dev\.of(\d+)runs,(\d+)loop(?:s)?eachmean ± std\. dev\. of (\d+) runs, (\d+) loop(?:s)? each"  # Fixed "loops" to "loop(?:s)?"
    )
    kv_pattern = re.compile(r"\t([^:]+):\s*(.*)")

    for line in lines:
        line = line.rstrip()
        header_match = header_pattern.match(line)
        if header_match:
            if current_record:
                records.append(current_record)
            current_record = {}
            current_record["name"] = header_match.group(1).strip()
            current_record["M"] = int(header_match.group(2))
            current_record["K"] = int(header_match.group(3))
            current_record["NUM_THREADS"] = int(header_match.group(4))
            continue

        if not current_record:
            continue

        timing_match = timing_pattern.match(line)
        if timing_match:
            time_val_str = timing_match.group(1)
            time_unit_str = timing_match.group(2)
            std_dev_val_str = timing_match.group(3)
            std_dev_unit_str = timing_match.group(4)

            s_str_for_timedelta = f"{time_val_str} {time_unit_str}".replace("µs", "us")
            ds_str_for_timedelta = f"{std_dev_val_str} {std_dev_unit_str}".replace("µs", "us")

            try:
                current_record["s"] = pd.to_timedelta(s_str_for_timedelta)
                current_record["ds"] = pd.to_timedelta(ds_str_for_timedelta)
            except ValueError as e:
                print(f"Warning: Could not parse time string '{s_str_for_timedelta}' or '{ds_str_for_timedelta}': {e}")
                current_record["s"] = pd.NaT
                current_record["ds"] = pd.NaT

            current_record["runs"] = int(timing_match.group(5))
            current_record["loops"] = int(timing_match.group(6))
            continue

        kv_match = kv_pattern.match(line)
        if kv_match:
            key = kv_match.group(1).strip()
            value_str = kv_match.group(2).strip()
            processed_value = None

            try:
                if value_str.endswith("%"):
                    processed_value = float(value_str[:-1])
                else:
                    processed_value = float(value_str)
                if processed_value.is_integer():
                    processed_value = int(processed_value)
            except ValueError:
                if len(value_str) >= 2 and value_str.startswith('"') and value_str.endswith('"'):
                    processed_value = value_str[1:-1]
                else:
                    processed_value = value_str
            current_record[key] = processed_value
            continue

    if current_record:
        records.append(current_record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if df.empty:
        return df

    preferred_order_start = ["name", "M", "K", "NUM_THREADS", "s", "ds", "runs", "loops"]
    specific_next_col = "Command being timed"

    final_columns = []
    present_columns = list(df.columns)

    for col in preferred_order_start:
        if col in present_columns:
            final_columns.append(col)
            present_columns.remove(col)

    if specific_next_col in present_columns:
        final_columns.append(specific_next_col)
        present_columns.remove(specific_next_col)

    present_columns.sort()
    final_columns.extend(present_columns)

    df = df.reindex(columns=final_columns)

    return df


# %%
with open("profile_8192.log", "r") as f:
    df = parse_log_to_dataframe(f.read())

# %%
df.set_index(["name", "M", "K", "NUM_THREADS"], inplace=True)
df["s"] = df["s"].dt.total_seconds()
df["ds"] = df["ds"].dt.total_seconds()

# %%
df[["s", "Percent of CPU this job got", "Maximum resident set size (kbytes)"]]

# %% [markdown]
# This shows the memory use, and only "fori" kinds does not scale with $K$.

# %%
(df["Maximum resident set size (kbytes)"] * 1e-6).groupby(level=(0, 2)).describe()

# %%
df["s"].droplevel(1).unstack((1, 2))

# %%
import plotly.express as px

# %%
df_plot = df[["s", "ds"]].droplevel(1).reset_index()
df_plot.head()

# %%
px.line(
    df_plot,
    x="NUM_THREADS",  # or whatever you want on x-axis
    y="s",  # your Series values
    error_y="ds",
    color="name",
    facet_col="K",
    log_y=True,
)
