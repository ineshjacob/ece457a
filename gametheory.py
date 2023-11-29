import pandas as pd
from nashpy import Game

# Define the strategies for the Democratic and Republican parties
democratic_strategies = ["Taxing the Rich", "Joining Paris Climate Agreement", "Abortion should be Legal", "Affordable Healthcare"]
republican_strategies = ["Tax Cuts", "Leaving Paris Climate Agreement", "Abortion should be illegal", "Privatized Healthcare"]

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to create the payoff matrix for one state
def create_state_payoff_matrix(state, education_data, gdp_data, abortion_data, healthcare_data):
    payoff_matrix = pd.DataFrame(0.0, index=democratic_strategies, columns=republican_strategies)
    gdp_preference = gdp_data[gdp_data['State'] == state]['Normalized GDP'].iloc[0]
    education_preference = education_data[education_data['State'] == state]['Normalized DegreePerCapita'].iloc[0]
    abortion_preference = abortion_data[abortion_data['state'] == state]['Normalized Abortion Rate'].iloc[0]
    healthcare_preference = healthcare_data[healthcare_data['State'] == state]['Normalized Spending'].iloc[0]

    payoff_matrix.at["Taxing the Rich", "Tax Cuts"] = gdp_preference
    payoff_matrix.at["Joining Paris Climate Agreement", "Leaving Paris Climate Agreement"] = education_preference
    payoff_matrix.at["Abortion should be Legal", "Abortion should be illegal"] = abortion_preference
    payoff_matrix.at["Affordable Healthcare", "Privatized Healthcare"] = healthcare_preference

    return payoff_matrix

# Function to calculate the Nash equilibrium
def calculate_nash_equilibrium(payoff_matrix):
    game = Game(payoff_matrix.values, payoff_matrix.values.T)
    equilibria = game.support_enumeration()
    return list(equilibria)

# Function to format Nash equilibria output in a readable format
def format_nash_equilibria(nash_equilibria, democratic_strategies, republican_strategies):
    formatted_output = ""
    for equilibrium in nash_equilibria:
        dem_eq, rep_eq = equilibrium
        formatted_output += "Democratic Strategies: \n"
        for i, prob in enumerate(dem_eq):
            if prob > 0:
                formatted_output += f"  {democratic_strategies[i]}: {prob:.2f}\n"
        formatted_output += "Republican Strategies: \n"
        for i, prob in enumerate(rep_eq):
            if prob > 0:
                formatted_output += f"  {republican_strategies[i]}: {prob:.2f}\n"
        formatted_output += "\n"
    return formatted_output

def find_dominant_strategies(payoff_matrix):
    dominant_strategies = {'Democratic': None, 'Republican': None}

    # Check for dominant strategies for the Democratic player
    for dem_strategy in democratic_strategies:
        is_dominant = True
        dem_payoffs = payoff_matrix.loc[dem_strategy]  # Payoffs when Democratic player chooses dem_strategy
        for other_dem_strategy in democratic_strategies:
            if dem_strategy != other_dem_strategy:
                other_payoffs = payoff_matrix.loc[other_dem_strategy]  # Payoffs for other Democratic strategies
                if (dem_payoffs < other_payoffs).any():
                    is_dominant = False
                    break
        if is_dominant:
            dominant_strategies['Democratic'] = dem_strategy
            break

    # Check for dominant strategies for the Republican player
    for rep_strategy in republican_strategies:
        is_dominant = True
        rep_payoffs = payoff_matrix[rep_strategy]  # Payoffs when Republican player chooses rep_strategy
        for other_rep_strategy in republican_strategies:
            if rep_strategy != other_rep_strategy:
                other_payoffs = payoff_matrix[other_rep_strategy]  # Payoffs for other Republican strategies
                if (rep_payoffs < other_payoffs).any():
                    is_dominant = False
                    break
        if is_dominant:
            dominant_strategies['Republican'] = rep_strategy
            break

    return dominant_strategies

# Load the processed data
abortion_data = load_data('processed_data/abortion_swing_state_data.csv')
education_data = load_data('processed_data/education_swing_state.csv')
gdp_data = load_data('processed_data/gdp_swing_state_data.csv')
healthcare_data = load_data('processed_data/healthcare_swing_state_data.csv')

# File writing setup for payoff matrices and strategies
payoff_matrices_content = ""
strategies_content = ""
nash_equilibria_content = ""

for state in education_data['State'].unique():
    payoff_matrix = create_state_payoff_matrix(state, education_data, gdp_data, abortion_data, healthcare_data)
    payoff_matrices_content += f"State: {state}\n"
    payoff_matrices_content += "Payoff Matrix:\n"
    payoff_matrices_content += payoff_matrix.to_string() + "\n\n"

     # Calculate and format Nash equilibria
    nash_equilibria = calculate_nash_equilibrium(payoff_matrix)
    nash_equilibria_content += f"State: {state}\n"
    formatted_equilibria = format_nash_equilibria(nash_equilibria, democratic_strategies, republican_strategies)
    nash_equilibria_content += formatted_equilibria
    
    # Calculate and write dominant strategies
    # You can comment out this section if not needed
    dominant_strategies = find_dominant_strategies(payoff_matrix)
    strategies_content += f"State: {state}\n"
    strategies_content += f"  Dominant Strategy for Democratic Player: {dominant_strategies['Democratic']}\n"
    strategies_content += f"  Dominant Strategy for Republican Player: {dominant_strategies['Republican']}\n\n"

# Write the payoff matrices content to its file
with open('payoff_matrices.txt', 'w') as file:
    file.write(payoff_matrices_content)

# Write the formatted Nash equilibria content to its file
with open('nash_equilibria.txt', 'w') as file:
    file.write(nash_equilibria_content)

# Write the dominant strategies content to its file
with open('dominant_strategies.txt', 'w') as file:
    file.write(strategies_content)
