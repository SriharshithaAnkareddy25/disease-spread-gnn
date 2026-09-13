"""Merge weekly COVID case data onto the 2011-census district node table.

Case data (data.incovid19.org, 2020-2023) uses present-day districts (781 of
them); the node table uses 2011 census districts (640). The gap is mostly
real administrative history, not typos: many states carved new districts
after 2011 (Telangana split from Andhra Pradesh in 2014, plus one-off splits
in nearly every state through 2016-2020), so a chunk of case-data districts
have no 2011 parent to fall back to without a hand-built historical
crosswalk (out of scope here). This script:
  1. exact-matches on normalized (state, district) name,
  2. fuzzy-matches the remainder within the same state (catches spelling
     variants like Haridwar/Hardwar),
  3. reports what's still unmatched instead of silently dropping it.
"""
import re
import difflib

import pandas as pd

from district_aliases import apply_aliases

CASES_PATH = "data/raw/covid_cases/districts.csv"
NODES_PATH = "data/processed/nodes.csv"

JUNK_DISTRICTS = {"UNKNOWN", "OTHER STATE", "OTHER STATE/", "OTHER REGION"}


def norm(s: str) -> str:
    s = str(s).upper().strip()
    s = re.sub(r"[.\-]", " ", s)
    s = s.replace("&", "AND")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_crosswalk(cases: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    case_pairs = cases[["state_norm", "district_norm"]].drop_duplicates()
    node_lookup = nodes.set_index(["state_norm", "district_norm"])["censuscode"].to_dict()
    nodes_by_state = nodes.groupby("state_norm")["district_norm"].apply(list).to_dict()

    rows = []
    for _, r in case_pairs.iterrows():
        state, district = r["state_norm"], r["district_norm"]
        if district in JUNK_DISTRICTS:
            continue
        censuscode = node_lookup.get((state, district))
        method = "exact" if censuscode is not None else None

        if censuscode is None:
            candidates = nodes_by_state.get(state, [])
            close = difflib.get_close_matches(district, candidates, n=1, cutoff=0.82)
            if close:
                censuscode = node_lookup[(state, close[0])]
                method = "fuzzy"

        rows.append({"state_norm": state, "district_norm": district, "censuscode": censuscode, "method": method})

    return pd.DataFrame(rows)


def main():
    cases = pd.read_csv(CASES_PATH, parse_dates=["Date"])
    nodes = pd.read_csv(NODES_PATH)

    cases["district_norm"] = cases["District"].apply(norm)
    cases["state_norm"] = cases["State"].apply(norm)
    cases["state_norm"], cases["district_norm"] = zip(
        *cases.apply(lambda r: apply_aliases(r["state_norm"], r["district_norm"]), axis=1)
    )
    nodes["district_norm"] = nodes["DISTRICT"].apply(norm)
    nodes["state_norm"] = nodes["ST_NM"].apply(norm)

    crosswalk = build_crosswalk(cases, nodes)
    matched = crosswalk.dropna(subset=["censuscode"])
    unmatched = crosswalk[crosswalk["censuscode"].isna()]

    print(f"Case-data (state,district) pairs: {len(crosswalk)}")
    print(f"  matched exact: {(matched['method'] == 'exact').sum()}")
    print(f"  matched fuzzy: {(matched['method'] == 'fuzzy').sum()}")
    print(f"  unmatched:     {len(unmatched)}")
    print(f"Node districts with >=1 matched case-data pair: "
          f"{matched['censuscode'].nunique()} / {len(nodes)}")

    unmatched.drop(columns=["censuscode", "method"]).to_csv(
        "data/processed/case_data_unmatched_districts.csv", index=False
    )

    cases = cases.merge(crosswalk[["state_norm", "district_norm", "censuscode"]], on=["state_norm", "district_norm"])
    cases = cases.dropna(subset=["censuscode"])
    cases["censuscode"] = cases["censuscode"].astype(int)

    # Multiple present-day districts can map to the same 2011 parent (splits) -
    # sum their counts per week under that parent's censuscode.
    cases["week"] = cases["Date"].dt.to_period("W-SUN").dt.start_time
    weekly = (
        cases.groupby(["censuscode", "week"])[["Confirmed", "Recovered", "Deceased", "Tested"]]
        .sum()
        .reset_index()
    )
    # Source data has rare negative values from upstream corrections/revisions
    # (e.g. a recovery count revised down) -- clip rather than let a
    # spatio-temporal model interpret them as a real negative-case week.
    count_cols = ["Confirmed", "Recovered", "Deceased", "Tested"]
    negative_count = (weekly[count_cols] < 0).sum().sum()
    if negative_count:
        print(f"Clipping {negative_count} negative value(s) in {count_cols} to 0")
        weekly[count_cols] = weekly[count_cols].clip(lower=0)

    covered_pop = nodes[nodes["censuscode"].isin(matched["censuscode"])]["Population"].sum()
    total_pop = nodes["Population"].sum()
    print(f"Population coverage: {covered_pop / total_pop:.1%}")

    weekly.to_csv("data/processed/weekly_case_features.csv", index=False)
    print(f"Saved data/processed/weekly_case_features.csv "
          f"({weekly['censuscode'].nunique()} districts, "
          f"{weekly['week'].nunique()} weeks, {len(weekly)} rows)")


if __name__ == "__main__":
    main()
