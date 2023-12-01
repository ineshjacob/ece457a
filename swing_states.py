import pandas as pd
import json
import os

presidential_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"raw_data/1976-2020-president.csv"))


def calculate_margin_of_victory(group):
    sorted_group = group.sort_values(by="candidatevotes", ascending=False)
    top_two = sorted_group.head(2)
    margin = (
        top_two.iloc[0]["candidatevotes"] - top_two.iloc[1]["candidatevotes"]
    ) / group["totalvotes"].sum()
    return margin


grouped = presidential_data.groupby(["year", "state_po"])
margins = grouped.apply(calculate_margin_of_victory).reset_index(name="margin")

winning_party = presidential_data.loc[
    presidential_data.groupby(["year", "state_po"])["candidatevotes"].idxmax()
]
margins = margins.merge(
    winning_party[["year", "state_po", "party_simplified"]], on=["year", "state_po"]
)

swing_states = {}
for (year, state), group in margins.groupby(["year", "state_po"]):
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
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"processed_data/swing_states.json"),'w') as f:
    json.dump(top_swing_states,f)
#git test 
