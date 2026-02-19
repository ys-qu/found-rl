"""
Extract evaluation metrics from WandB and compute Arrival Rate (redefined success rate)
and Strict Success Rate. Logs Full_Summary and Brief_Summary tables back to WandB.
"""
import wandb
import pandas as pd
import sys

WANDB_ENTITY = "qu120-purdue-university"
WANDB_PROJECT = "found_rl"
TARGET_GROUP = "offline_rl_new"  # Group name to filter
NAME_KEYWORD = "test"  # String that must be included in run name

# Metric configuration
WEIGHTED_KEYS = [
    'collisions_layout', 'collisions_vehicle', 'collisions_pedestrian',
    'collisions_others', 'red_light', 'stop_infraction', 'route_dev',
    'vehicle_blocked', 'percentage_outside_lane', 'percentage_wrong_lane'
]

BRIEF_KEYS = [
    "Strict Success Rate (%)", "Arrival Rate (%)", "Icell", "fuel_rate", "collisions_vehicle",
    "collisions_pedestrian", "red_light", "reward", "score_composed",
    "score_penalty", "score_route", "speed_norm"
]


def find_col(df, key):
    if key in df.columns: return key
    candidates = [c for c in df.columns if c.endswith(f"/{key}") or f"/{key}/" in c]
    return candidates[0] if len(candidates) == 1 else None


def calculate_metrics(run):
    """Calculation logic for a single run"""
    print(f"Processing: {run.name} ({run.id}) ...")

    # Get history data
    df = pd.DataFrame([row for row in run.scan_history()]).fillna(0)
    if df.empty:
        print(f"Skipping {run.name}: No history data")
        return None, None

    dist_col = find_col(df, 'route_completed_in_km')
    total_dist = df[dist_col].sum()
    score_col = find_col(df, 'score_route')

    # Calculate Strict Success Rate
    has_any_infraction = pd.Series([False] * len(df))
    for wk in WEIGHTED_KEYS:
        col = find_col(df, wk)
        if col: has_any_infraction |= ((df[col] * df[dist_col]) > 0.0001)

    ssr = ((df[score_col] >= 0.999) & (~has_any_infraction)).mean() * 100
    cr = (df[score_col] >= 0.999).mean() * 100

    # Fill Master Data
    master_data = {
        "Run Name": run.name,
        "Total Distance (km)": total_dist,
        "Arrival Rate (%)": cr,
        "Strict Success Rate (%)": ssr
    }

    hero_cols = [c for c in df.columns if c.startswith("hero/")]
    for col in hero_cols:
        clean_name = col.replace("hero/", "")
        if clean_name in WEIGHTED_KEYS:
            val = (df[col] * df[dist_col]).sum() / total_dist if total_dist > 0 else 0
        else:
            val = df[col].mean()
        master_data[clean_name] = val

    full_df = pd.DataFrame([master_data])
    brief_data = {}
    for bk in BRIEF_KEYS:
        if bk in master_data:
            brief_data[bk] = master_data[bk]
        elif bk == "Strict Success Rate (%)":
            brief_data[bk] = ssr
        elif bk == "Arrival Rate (%)":
            brief_data[bk] = cr

    brief_df = pd.DataFrame([brief_data])

    return full_df, brief_df


def main():
    api = wandb.Api()

    filters = {
        "group": TARGET_GROUP,
        "display_name": {"$regex": NAME_KEYWORD}
    }

    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters)
    print(f"Found {len(runs)} runs matching criteria")

    for run in runs:
        full_df, brief_df = calculate_metrics(run)

        if full_df is not None:
            print(f"\n--- {run.name} Summary Results ---")
            print(brief_df)
            # print(full_df)

            with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, id=run.id, resume="allow") as r:
                r.log({
                    "table/Full_Summary": wandb.Table(dataframe=full_df),
                    "table/Brief_Summary": wandb.Table(dataframe=brief_df)
                })
            print(f"Successfully updated WandB Table for Run: {run.name}")


if __name__ == "__main__":
    main()