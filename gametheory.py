import pandas as pd
from nashpy import Game

# Define the strategies for the Democratic and Republican parties
democratic_strategies = ["Taxing the Rich", "Joining Paris Climate Agreement", 
                         "Abortion should be Legal", "Affordable Healthcare"]
republican_strategies = ["Tax Cuts", "Leaving Paris Climate Agreement", 
                         "Abortion should be illegal", "Privatized Healthcare"]

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to create the payoff matrix
def create_payoff_matrix(education_data, gdp_data, abortion_data, healthcare_data):
    # Initialize the payoff matrix with zeros of float type
    payoff_matrix = pd.DataFrame(0.0, index=democratic_strategies, columns=republican_strategies)
    
    # Assuming each state's data is already normalized and represents the preference for each policy
    for state in education_data['State'].unique():
        # Extract the normalized values for each policy for the state
        education_preference = education_data[education_data['State'] == state]['Normalized DegreePerCapita'].values[0]
        gdp_preference = gdp_data[gdp_data['State'] == state]['Smoothed GDP'].values[0]
        abortion_preference = abortion_data[abortion_data['state'] == state]['Normalized Abortion Rate'].values[0]
        healthcare_preference = healthcare_data[healthcare_data['State'] == state]['Normalized Spending'].values[0]
        
        # Update the payoff matrix with the preferences
        payoff_matrix.at["Joining Paris Climate Agreement", "Leaving Paris Climate Agreement"] = float(education_preference)
        payoff_matrix.at["Taxing the Rich", "Tax Cuts"] = float(gdp_preference)
        payoff_matrix.at["Abortion should be Legal", "Abortion should be illegal"] = float(abortion_preference)
        payoff_matrix.at["Affordable Healthcare", "Privatized Healthcare"] = float(healthcare_preference)
    
    print(payoff_matrix)
    return payoff_matrix

# Function to calculate the Nash equilibrium
def calculate_nash_equilibrium(payoff_matrix):
    game = Game(payoff_matrix.values, payoff_matrix.values.T)
    equilibria = game.support_enumeration()
    return list(equilibria)

# Load the processed data
abortion_data = load_data('processed_data/abortion_swing_state_data.csv')
education_data = load_data('processed_data/education_swing_state.csv')
gdp_data = load_data('processed_data/gdp_swing_state_data.csv')
healthcare_data = load_data('processed_data/healthcare_swing_state_data.csv')

# Create the payoff matrix
payoff_matrix = create_payoff_matrix(education_data, gdp_data, abortion_data, healthcare_data)

# Calculate the Nash equilibrium
nash_equilibria = calculate_nash_equilibrium(payoff_matrix)

# Print the Nash equilibria
#for eq in nash_equilibria:
    #print(eq)


# First, let's define a function that will format and print the Nash equilibria in a more readable way.
def format_nash_equilibria(nash_equilibria, democratic_strategies, republican_strategies):
    """
    Format and print the Nash equilibria in a human-readable way.
    
    Parameters:
    - nash_equilibria: A list of tuples representing the strategy distributions in Nash equilibria.
    - democratic_strategies: The list of strategies for the Democratic player.
    - republican_strategies: The list of strategies for the Republican player.
    """
    readable_output = []

    for equilibrium in nash_equilibria:
        dem_strategies_probs, rep_strategies_probs = equilibrium
        # Create a formatted string for each player's strategy distribution
        dem_distribution = ", ".join(
            f"{democratic_strategies[i]}: {prob:.2%}"
            for i, prob in enumerate(dem_strategies_probs) if prob > 0
        )
        rep_distribution = ", ".join(
            f"{republican_strategies[i]}: {prob:.2%}"
            for i, prob in enumerate(rep_strategies_probs) if prob > 0
        )
        readable_output.append(f"Democratic Player: {dem_distribution}\nRepublican Player: {rep_distribution}\n")
    
    return "\n".join(readable_output)

# Use the function to format the Nash equilibria from the calculated results
formatted_output = format_nash_equilibria(nash_equilibria, democratic_strategies, republican_strategies)

print(formatted_output)


