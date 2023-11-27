import pandas as pd


abortion_data = pd.read_csv("raw_data/NationalAndStatePregnancy_PublicUse.csv")

states_list = ["OH", "FL", "WI", "IA", "PA", "GA", "CO", "MI", "NM", "LA"]

metric = "abortionratetotal"

filtered_abortion_data = abortion_data[abortion_data["state"].isin(states_list)]

filtered_abortion_data["Average Abortion Rate"] = filtered_abortion_data.groupby(
    "state"
)[metric].transform("mean")

Q1 = filtered_abortion_data["Average Abortion Rate"].quantile(0.25)
Q3 = filtered_abortion_data["Average Abortion Rate"].quantile(0.75)
IQR = Q3 - Q1


def normalize_value(val, Q1, Q3, IQR):
    if val < Q1:
        return 0.05 + 0.20 * (
            (val - filtered_abortion_data["Average Abortion Rate"].min())
            / (Q1 - filtered_abortion_data["Average Abortion Rate"].min())
        )
    elif val > Q3:
        return 0.75 + 0.20 * (
            (val - Q3) / (filtered_abortion_data["Average Abortion Rate"].max() - Q3)
        )
    else:
        return 0.25 + 0.50 * ((val - Q1) / IQR)


filtered_abortion_data["Normalized Abortion Rate"] = filtered_abortion_data[
    "Average Abortion Rate"
].apply(lambda x: normalize_value(x, Q1, Q3, IQR))

final_abortion_data = (
    filtered_abortion_data[
        ["state", "Average Abortion Rate", "Normalized Abortion Rate"]
    ]
    .drop_duplicates(subset="state")
    .reset_index(drop=True)
)

final_abortion_data.to_csv("processed_data/abortion_swing_state_data.csv", index=False)
