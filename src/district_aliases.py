"""Manually-compiled aliases reconciling present-day case-data districts back to
their 2011-census parent district. Needed because entire states were
reorganized after the 2011 census (Telangana split from Andhra Pradesh in
2014, then Telangana itself carved 10 districts into 33 through 2016-2019;
Ladakh split from Jammu & Kashmir in 2019; several Karnataka districts were
officially renamed in 2014).

Telangana mapping is compiled from Wikipedia district articles. Splits for
Adilabad, Karimnagar, Medak, Nizamabad, Rangareddy and Warangal are
confirmed directly from each district's own article (each explicitly states
what it was divided into in October 2016). Khammam, Mahbubnagar and Nalgonda
splits are NOT confirmed from a single fetchable source — they're filled in
from well-documented public record (widely repeated across Indian civics
references) since Wikipedia's own articles for those three don't state their
split explicitly in fetchable form. Treat those three with more caution than
the other six if precision matters downstream.
"""

# new Telangana district (normalized) -> its 2011-census parent district (normalized, in Andhra Pradesh)
TELANGANA_TO_AP_PARENT = {
    # Adilabad split -- confirmed (Adilabad district Wikipedia article)
    "ADILABAD": "ADILABAD",
    "KOMARAM BHEEM": "ADILABAD",
    "MANCHERIAL": "ADILABAD",
    "NIRMAL": "ADILABAD",
    # Karimnagar split -- confirmed (Karimnagar district Wikipedia article)
    "KARIMNAGAR": "KARIMNAGAR",
    "JAGTIAL": "KARIMNAGAR",
    "PEDDAPALLI": "KARIMNAGAR",
    "RAJANNA SIRCILLA": "KARIMNAGAR",
    # Khammam split -- NOT independently confirmed; public record
    "KHAMMAM": "KHAMMAM",
    "BHADRADRI KOTHAGUDEM": "KHAMMAM",
    # Mahbubnagar split -- Nagarkurnool confirmed (its own Wikipedia article);
    # Wanaparthy/Jogulamba Gadwal/Narayanpet NOT independently confirmed, public record
    "MAHABUBNAGAR": "MAHBUBNAGAR",
    "NAGARKURNOOL": "MAHBUBNAGAR",
    "WANAPARTHY": "MAHBUBNAGAR",
    "JOGULAMBA GADWAL": "MAHBUBNAGAR",
    "NARAYANPET": "MAHBUBNAGAR",
    # Medak split -- confirmed (Medak district Wikipedia article)
    "MEDAK": "MEDAK",
    "SANGAREDDY": "MEDAK",
    "SIDDIPET": "MEDAK",
    # Nalgonda split -- NOT independently confirmed; public record
    "NALGONDA": "NALGONDA",
    "SURYAPET": "NALGONDA",
    "YADADRI BHUVANAGIRI": "NALGONDA",
    # Nizamabad split -- confirmed (Nizamabad district Wikipedia article)
    "NIZAMABAD": "NIZAMABAD",
    "KAMAREDDY": "NIZAMABAD",
    # Rangareddy split -- confirmed (Rangareddy district Wikipedia article)
    "RANGA REDDY": "RANGAREDDY",
    "VIKARABAD": "RANGAREDDY",
    "MEDCHAL MALKAJGIRI": "RANGAREDDY",
    # Warangal split -- confirmed (Warangal district Wikipedia article); Mulugu
    # (2019 addendum, carved from Warangal Rural + Jayashankar Bhupalpally) NOT
    # independently confirmed, public record
    "WARANGAL URBAN": "WARANGAL",
    "WARANGAL RURAL": "WARANGAL",
    "JANGAON": "WARANGAL",
    "JAYASHANKAR BHUPALAPALLY": "WARANGAL",
    "MAHABUBABAD": "WARANGAL",
    "MULUGU": "WARANGAL",
    # Unchanged
    "HYDERABAD": "HYDERABAD",
}

# Officially renamed (2014) Karnataka districts: case-data name (normalized) -> 2011-census name (normalized)
KARNATAKA_RENAMES = {
    "BENGALURU URBAN": "BANGALORE",
    "BENGALURU RURAL": "BANGALORE RURAL",
    "BELAGAVI": "BELGAUM",
    "BALLARI": "BELLARY",
    "VIJAYAPURA": "BIJAPUR",
    "KALABURAGI": "GULBARGA",
    "MYSURU": "MYSORE",
    "CHAMARAJANAGAR": "CHAMRAJNAGAR",
}

# Ladakh split from Jammu & Kashmir in 2019; district boundaries unchanged.
LADAKH_TO_JK_DISTRICT = {
    "LEH": "LEH (LADAKH)",
    "KARGIL": "KARGIL",
}


def apply_aliases(state_norm: str, district_norm: str) -> tuple[str, str]:
    if state_norm == "TELANGANA" and district_norm in TELANGANA_TO_AP_PARENT:
        return "ANDHRA PRADESH", TELANGANA_TO_AP_PARENT[district_norm]
    if state_norm == "KARNATAKA" and district_norm in KARNATAKA_RENAMES:
        return state_norm, KARNATAKA_RENAMES[district_norm]
    if state_norm == "LADAKH" and district_norm in LADAKH_TO_JK_DISTRICT:
        return "JAMMU AND KASHMIR", LADAKH_TO_JK_DISTRICT[district_norm]
    return state_norm, district_norm
