import pandas as pd

df = pd.read_csv("raw_data/education_census.csv")

population_df = df.iloc[[1, 6, 16, 19, 22, 25]]
bachelor_degree_df = df.iloc[[5, 15, 18, 21, 24, 27]]

degree_relevant_columns = [
    col
    for col in list(bachelor_degree_df.columns)
    if "Total!!Estimate" in col and "!!Male!!" not in col and "!!Female!!" not in col
]
degree_relevant_df = bachelor_degree_df[degree_relevant_columns]

pop_relevant_columns = [
    col
    for col in list(population_df.columns)
    if "Total!!Estimate" in col and "!!Male!!" not in col and "!!Female!!" not in col
]
pop_relevant_df = population_df[pop_relevant_columns]


degree_relevant_df.columns = [col.split("!!")[0] for col in degree_relevant_columns]
pop_relevant_df.columns = [col.split("!!")[0] for col in pop_relevant_columns]

for col in degree_relevant_df.columns:
    degree_relevant_df[col] = pd.to_numeric(
        degree_relevant_df[col].str.replace(",", ""), errors="coerce"
    )
for col in pop_relevant_df.columns:
    pop_relevant_df[col] = pd.to_numeric(
        pop_relevant_df[col].str.replace(",", ""), errors="coerce"
    )

degree_totals = degree_relevant_df.sum()
pop_totals = pop_relevant_df.sum()
education_data = pd.DataFrame(
    {"BachelorDegrees": degree_totals, "Population": pop_totals}
)
education_data["DegreePerCapita"] = (
    education_data["BachelorDegrees"] / education_data["Population"]
)

# Calculate the IQR
Q1 = education_data["DegreePerCapita"].quantile(0.25)
Q3 = education_data["DegreePerCapita"].quantile(0.75)
IQR = Q3 - Q1


# Define the normalization function
def normalize_value(val, Q1, Q3, IQR):
    lower_bound = 0.05
    upper_bound = 0.95
    if val < Q1:
        return lower_bound + (0.25 - lower_bound) * (
            (val - education_data["DegreePerCapita"].min())
            / (Q1 - education_data["DegreePerCapita"].min())
        )
    elif val > Q3:
        return 0.75 + (upper_bound - 0.75) * (
            (val - Q3) / (education_data["DegreePerCapita"].max() - Q3)
        )
    else:
        return 0.25 + 0.50 * ((val - Q1) / IQR)


education_data["Normalized DegreePerCapita"] = education_data["DegreePerCapita"].apply(
    lambda x: normalize_value(x, Q1, Q3, IQR)
)

education_data[["DegreePerCapita", "Normalized DegreePerCapita"]].to_csv(
    "processed_data/education_swing_state.csv", index=True, index_label="State"
)
