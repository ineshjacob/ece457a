import pandas as pd

df = pd.read_csv(
    "raw_data/IHME_USA_STATE_HEALTH_SPENDING_2003_2019_DATA_Y2022M08D01.CSV"
)

states_list = [
    "Ohio",
    "Florida",
    "Wisconsin",
    "Iowa",
    "Pennsylvania",
    "Georgia",
    "Colorado",
    "Michigan",
    "New Mexico",
    "Louisiana",
]

df = df[df["state"].isin(states_list)]

df["Average Spending"] = df.groupby("state")["val"].transform("mean")

Q1 = df["Average Spending"].quantile(0.25)
Q3 = df["Average Spending"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = 0.05
upper_bound = 0.95


def normalize_value(val, Q1, Q3, IQR, lower_bound, upper_bound):
    if val < Q1:
        # Scale values below Q1 to lower_bound - 0.25 range
        return lower_bound + (0.25 - lower_bound) * (
            (val - df["Average Spending"].min()) / (Q1 - df["Average Spending"].min())
        )
    elif val > Q3:
        # Scale values above Q3 to 0.75 - upper_bound range
        return 0.75 + (upper_bound - 0.75) * (
            (val - Q3) / (df["Average Spending"].max() - Q3)
        )
    else:
        # Scale values within IQR to 0.25-0.75 range
        return 0.25 + 0.50 * ((val - Q1) / IQR)


df["Normalized Spending"] = df["Average Spending"].apply(
    lambda x: normalize_value(x, Q1, Q3, IQR, lower_bound, upper_bound)
)

final_data = (
    df[["state", "Average Spending", "Normalized Spending"]]
    .drop_duplicates(subset="state")
    .reset_index(drop=True)
)

final_data.to_csv("processed_data/healthcare_swing_state_data.csv", index=False)
