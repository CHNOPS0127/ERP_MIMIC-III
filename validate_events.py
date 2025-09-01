import os
import argparse
import pandas as pd
from tqdm import tqdm


def is_subject_folder(x):
    """Return True if folder name is numeric (a SUBJECT_ID)."""
    return str.isdigit(x)


def main():
    """Clean per-subject events.csv using stays.csv as reference."""

    # Counters
    n_events = 0                   # total events
    empty_hadm = 0                 # events with empty HADM_ID (drop)
    no_hadm_in_stay = 0            # HADM_ID not found in stays.csv (drop)
    no_icustay = 0                 # events with empty ICUSTAY_ID (try to fill)
    recovered = 0                  # ICUSTAY_ID filled from stays.csv
    could_not_recover = 0          # ICUSTAY_ID still missing after fill (should be 0)
    icustay_missing_in_stays = 0   # ICUSTAY_ID mismatch vs stays.csv (drop)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "subjects_root_path",
        type=str,
        help="Directory containing subject subdirectories."
    )
    args = parser.parse_args()
    print(args)

    subdirectories = os.listdir(args.subjects_root_path)
    subjects = list(filter(is_subject_folder, subdirectories))

    for subject in tqdm(subjects, desc="Iterating over subjects"):
        # Load stays; ensure HADM_ID/ICUSTAY_ID are strings
        stays_df = pd.read_csv(
            os.path.join(args.subjects_root_path, subject, "stays.csv"),
            index_col=False,
            dtype={"HADM_ID": str, "ICUSTAY_ID": str}
        )
        stays_df.columns = stays_df.columns.str.upper()

        # Sanity checks: no missing IDs, and each ID appears once
        assert not stays_df["ICUSTAY_ID"].isnull().any()
        assert not stays_df["HADM_ID"].isnull().any()
        assert len(stays_df["ICUSTAY_ID"].unique()) == len(stays_df["ICUSTAY_ID"])
        assert len(stays_df["HADM_ID"].unique()) == len(stays_df["HADM_ID"])

        # Load events
        events_df = pd.read_csv(
            os.path.join(args.subjects_root_path, subject, "events.csv"),
            index_col=False,
            dtype={"HADM_ID": str, "ICUSTAY_ID": str}
        )
        events_df.columns = events_df.columns.str.upper()
        n_events += events_df.shape[0]

        # Drop events with empty HADM_ID
        # (Optional TODO: recover using ICUSTAY_ID)
        empty_hadm += events_df["HADM_ID"].isnull().sum()
        events_df = events_df.dropna(subset=["HADM_ID"])

        # Join events to stays on HADM_ID
        merged_df = events_df.merge(
            stays_df,
            left_on=["HADM_ID"],
            right_on=["HADM_ID"],
            how="left",
            suffixes=["", "_r"],
            indicator=True,
        )

        # Keep only events whose HADM_ID exists in stays.csv
        no_hadm_in_stay += (merged_df["_merge"] == "left_only").sum()
        merged_df = merged_df[merged_df["_merge"] == "both"]

        # Fill missing ICUSTAY_ID from stays (ICUSTAY_ID_r)
        cur_no_icustay = merged_df["ICUSTAY_ID"].isnull().sum()
        no_icustay += cur_no_icustay
        merged_df.loc[:, "ICUSTAY_ID"] = merged_df["ICUSTAY_ID"].fillna(merged_df["ICUSTAY_ID_r"])
        recovered += cur_no_icustay - merged_df["ICUSTAY_ID"].isnull().sum()
        could_not_recover += merged_df["ICUSTAY_ID"].isnull().sum()
        merged_df = merged_df.dropna(subset=["ICUSTAY_ID"])

        # Drop events where ICUSTAY_ID (events) != ICUSTAY_ID_r (stays)
        icustay_missing_in_stays += (merged_df["ICUSTAY_ID"] != merged_df["ICUSTAY_ID_r"]).sum()
        merged_df = merged_df[merged_df["ICUSTAY_ID"] == merged_df["ICUSTAY_ID_r"]]

        # Save cleaned events
        to_write = merged_df[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE", "VALUEUOM"]]
        to_write.to_csv(os.path.join(args.subjects_root_path, subject, "events.csv"), index=False)

    # All ICUSTAY_ID should be recovered
    assert could_not_recover == 0

    # Summary
    print(f"n_events: {n_events}")
    print(f"empty_hadm: {empty_hadm}")
    print(f"no_hadm_in_stay: {no_hadm_in_stay}")
    print(f"no_icustay: {no_icustay}")
    print(f"recovered: {recovered}")
    print(f"could_not_recover: {could_not_recover}")
    print(f"icustay_missing_in_stays: {icustay_missing_in_stays}")


if __name__ == "__main__":
    main()
