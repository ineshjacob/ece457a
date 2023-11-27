import pandas as pd

df = pd.read_csv("raw_data/education_census.csv")

population_18_24_df = df[
    df["Label (Grouping)"].str.contains("Population 18 to 24 years", na=False)
]

population_18_24_df = population_18_24_df.drop(columns=["Label (Grouping)"])

# Transposing the DataFrame for better readability
population_18_24_df = population_18_24_df.transpose()
population_18_24_df.columns = ["Total Estimate 18-24"]
population_18_24_df.reset_index(inplace=True)
population_18_24_df.rename(columns={"index": "State"}, inplace=True)


total_estimate_columns = [col for col in df.columns if "Total!!Estimate" in col]


# filtering the DataFrame for the row corresponding to "Population 25 years and over"
population_25_over_df = df[
    df["Label (Grouping)"].str.contains("Population 25 years and over", na=False)
]

# Selecting the first row if there are multiple (assuming it's the total estimate)
population_25_over_row = population_25_over_df.iloc[0]

state_population_25_over_estimates = {}
for col in total_estimate_columns:
    state_name = col.split("!!")[0]
    estimate = population_25_over_row[col]
    if pd.notna(estimate):  # Check if the value is not NaN
        estimate = int(estimate.replace(",", ""))
    state_population_25_over_estimates[state_name] = estimate


state_population_25_over_estimates_df = pd.DataFrame(
    list(state_population_25_over_estimates.items()),
    columns=["State", "Total Estimate"],
)

# Calculate the IQR of the 'Total Estimate'
Q1 = state_population_25_over_estimates_df["Total Estimate"].quantile(0.25)
Q3 = state_population_25_over_estimates_df["Total Estimate"].quantile(0.75)
IQR = Q3 - Q1

# Define a function to normalize values based on IQR with smoothing
lower_bound = 0.05
upper_bound = 0.95


def normalize_value(val, Q1, Q3, IQR, lower_bound, upper_bound):
    if val < Q1:
        # Scale values below Q1 to lower_bound - 0.25 range
        return lower_bound + (0.25 - lower_bound) * (
            (val - state_population_25_over_estimates_df["Total Estimate"].min())
            / (Q1 - state_population_25_over_estimates_df["Total Estimate"].min())
        )
    elif val > Q3:
        # Scale values above Q3 to 0.75 - upper_bound range
        return 0.75 + (upper_bound - 0.75) * (
            (val - Q3)
            / (state_population_25_over_estimates_df["Total Estimate"].max() - Q3)
        )
    else:
        # Scale values within IQR to 0.25-0.75 range
        return 0.25 + 0.50 * ((val - Q1) / IQR)


# Apply the smoothing normalization to the 'Total Estimate' column
state_population_25_over_estimates_df[
    "Normalized Estimate"
] = state_population_25_over_estimates_df["Total Estimate"].apply(
    lambda x: normalize_value(x, Q1, Q3, IQR, lower_bound, upper_bound)
)

state_population_25_over_estimates_df[
    ["State", "Total Estimate", "Normalized Estimate"]
].to_csv("processed_data/education_swing_state.csv", index=False)
