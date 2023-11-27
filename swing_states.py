import pandas as pd
import json

presidential_data = pd.read_csv("raw_data/1976-2020-president.csv")


def calculate_margin_of_victory(group):
    sorted_group = group.sort_values(by="candidatevotes", ascending=False)
    top_two = sorted_group.head(2)
    margin = (
        top_two.iloc[0]["candidatevotes"] - top_two.iloc[1]["candidatevotes"]
    ) / group["totalvotes"].sum()
    return margin


grouped = presidential_data.groupby(["year", "state"])
margins = grouped.apply(calculate_margin_of_victory).reset_index(name="margin")

winning_party = presidential_data.loc[
    presidential_data.groupby(["year", "state"])["candidatevotes"].idxmax()
]
margins = margins.merge(
    winning_party[["year", "state", "party_simplified"]], on=["year", "state"]
)

swing_states = {}
for (year, state), group in margins.groupby(["year", "state"]):
    if state not in swing_states:
        swing_states[state] = {
            "party_changes": 0,
            "average_margin": [],
            "last_party": None,
        }
    current_party = group["party_simplified"].values[0]
    if (
        swing_states[state]["last_party"]
        and swing_states[state]["last_party"] != current_party
    ):
        swing_states[state]["party_changes"] += 1
    swing_states[state]["average_margin"].append(group["margin"].values[0])
    swing_states[state]["last_party"] = current_party


for state in swing_states:
    swing_states[state]["average_margin"] = sum(
        swing_states[state]["average_margin"]
    ) / len(swing_states[state]["average_margin"])
sorted_states = sorted(
    swing_states.items(),
    key=lambda x: (x[1]["party_changes"], -x[1]["average_margin"]),
    reverse=True,
)

top_swing_states = sorted_states[:10]
print(json.dumps(top_swing_states, indent=2))
