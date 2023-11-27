import pandas as pd

df = pd.read_csv("raw_data/SAGDP1__ALL_AREAS_2017_2022.csv")

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

df = df[df["GeoName"].isin(states_list)]

df = df[df["Description"] == "Real GDP (millions of chained 2017 dollars) 1/"]
df = df[["GeoName", "2017", "2018", "2019", "2020", "2021", "2022"]]

df.reset_index(drop=True, inplace=True)
gdp_columns = df.columns[1:]


df["Average GDP"] = df.iloc[:, 1:].mean(axis=1)
# calculate the IQR of the average GDP
Q1 = df["Average GDP"].quantile(0.25)
Q3 = df["Average GDP"].quantile(0.75)
IQR = Q3 - Q1


# define new thresholds for smoothing
lower_bound = 0.05
upper_bound = 0.95


def normalize_value_smooth(val, Q1, Q3, IQR, lower_bound, upper_bound):
    if val < Q1:
        # scale values below Q1 to lower_bound - 0.25 range
        return lower_bound + (0.25 - lower_bound) * (
            (val - df["Average GDP"].min()) / (Q1 - df["Average GDP"].min())
        )
    elif val > Q3:
        # scale values above Q3 to 0.75 - upper_bound range
        return 0.75 + (upper_bound - 0.75) * (
            (val - Q3) / (df["Average GDP"].max() - Q3)
        )
    else:
        # scale values within IQR to 0.25-0.75 range
        return 0.25 + 0.5 * ((val - Q1) / IQR)


df["Smoothed GDP"] = df["Average GDP"].apply(
    lambda x: normalize_value_smooth(x, Q1, Q3, IQR, lower_bound, upper_bound)
)

df[["GeoName", "Average GDP", "Smoothed GDP"]].to_csv(
    "processed_data/gdp_swing_state_data.csv", index=False
)
