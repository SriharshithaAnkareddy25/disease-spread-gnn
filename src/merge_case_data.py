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
COUNT_COLS = ["Confirmed", "Recovered", "Deceased", "Tested"]


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

    weekly = weekly_cumulative(cases)
    # Increments only where both this week and the previous one are observed;
    # after a gap the diff would lump several weeks together.
    prev_observed = weekly.groupby("censuscode")["observed"].shift(fill_value=False)
    for col in COUNT_COLS:
        new = weekly.groupby("censuscode")[col].diff().clip(lower=0)
        weekly[f"new_{col.lower()}"] = new.where(prev_observed)
    usable = weekly.loc[weekly["observed"], "censuscode"].unique()
    print(f"Districts with usable case data: {len(usable)}")

    covered_pop = nodes[nodes["censuscode"].isin(usable)]["Population"].sum()
    total_pop = nodes["Population"].sum()
    print(f"Population coverage: {covered_pop / total_pop:.1%}")

    active = weekly[weekly["new_confirmed"] > 0].groupby("week")["censuscode"].nunique()
    print("Districts with new confirmed cases, last weeks with >=50% active: "
          f"{active[active >= 0.5 * weekly['censuscode'].nunique()].index.max().date()}")

    weekly.to_csv("data/processed/weekly_case_features.csv", index=False)
    print(f"Saved data/processed/weekly_case_features.csv "
          f"({weekly['censuscode'].nunique()} districts, "
          f"{weekly['week'].nunique()} weeks, {len(weekly)} rows, "
          f"{weekly['observed'].mean():.1%} observed)")


def clean_cumulative(s: pd.Series) -> pd.Series:
    """Clean one source district's daily cumulative series (indexed by Date).

    - A 0 after a nonzero value means the district stopped reporting (e.g. UP
      dropped district bulletins on 2021-10-31), not that cases went to 0.
    - A centred 5-day rolling median removes one-day spikes/dips.
    - A reverse running minimum then enforces monotonicity, so remaining
      downward revisions are pushed back rather than becoming negative weeks.
    """
    s = s.where(~((s == 0) & (s.cummax() > 0))).dropna()
    if s.empty:
        return s
    s = s.rolling(5, center=True, min_periods=1).median()
    return s[::-1].cummin()[::-1]


def weekly_cumulative(cases: pd.DataFrame) -> pd.DataFrame:
    """Week-end cumulative counts per censuscode, on a full district x week grid.

    The source is cumulative, so each week takes the last reported value
    (summing daily cumulatives inflated counts ~7x and made partial weeks look
    like drops). Multiple present-day districts can map to the same 2011
    parent (splits); their week-end cumulatives are summed. A parent week is
    `observed` only when it falls inside every child's reporting span -
    otherwise a child that stopped reporting would make the sum fall.
    """
    cases = cases.sort_values("Date")
    # Full weekly range: the source has no rows at all for 2022-10-17..23.
    weeks = pd.date_range(
        cases["Date"].min().to_period("W-SUN").start_time,
        cases["Date"].max().to_period("W-SUN").start_time,
        freq="W-MON",
    )

    child_frames = []
    for (code, state, district), g in cases.groupby(["censuscode", "state_norm", "district_norm"]):
        cols, last_change = {}, None
        for col in COUNT_COLS:
            daily = clean_cumulative(g.groupby("Date")[col].max())
            cols[col] = daily.groupby(daily.index.to_period("W-SUN").start_time).last()
            if col == "Confirmed" and not daily.empty:
                last_change = daily.index[daily.diff() > 0].max()
        child = pd.DataFrame(cols).reindex(weeks)
        valid = child["Confirmed"].notna()
        # All-zero rows (e.g. Assam, Telangana, Goa report under state-level
        # "Unknown") carry no district information at all.
        if not (child["Confirmed"] > 0).any():
            continue
        first, last = child.index[valid.argmax()], child.index[len(valid) - 1 - valid[::-1].argmax()]
        # From late 2022 the source stops updating district rows and only
        # carries the last value forward (all 2023 cases are filed under
        # state-level "Unknown"). Weeks after a district's final increase are
        # stale, not genuine zero-case weeks.
        if pd.notna(last_change):
            last = min(last, last_change.to_period("W-SUN").start_time)
        child = child.ffill()
        child["in_span"] = (child.index >= first) & (child.index <= last) & valid
        child["started"] = child.index >= first
        child[COUNT_COLS] = child[COUNT_COLS].fillna(0)
        child["censuscode"] = code
        child_frames.append(child.rename_axis("week").reset_index())

    children = pd.concat(child_frames)
    # A child that hasn't started yet is fine (its cases were still reported
    # under the pre-split name); one that has stopped is not.
    children["ok"] = children["in_span"] | ~children["started"]
    weekly = children.groupby(["censuscode", "week"]).agg(
        **{col: (col, "sum") for col in COUNT_COLS},
        observed=("ok", "all"),
        any_started=("started", "any"),
    ).reset_index()
    weekly["observed"] &= weekly["any_started"]
    weekly.loc[~weekly["observed"], COUNT_COLS] = float("nan")
    # Tested is 0 wherever a state never published tests; keep it NaN there.
    weekly.loc[weekly["Tested"] == 0, "Tested"] = float("nan")
    return weekly.drop(columns="any_started")


if __name__ == "__main__":
    main()
