import pandas as pd
from nashpy import Game
from fuzzy import get_data, calculate_payoff
import numpy as np

# Define the strategies for the Democratic and Republican parties
dem_strat = ["Abortion should be Legal",
                         "Joining Paris Climate Agreement", "Taxing the Rich",  "Affordable Healthcare"]
rep_strat = ["Abortion should be illegal",
                         "Leaving Paris Climate Agreement", "Tax Cuts", "Privatized Healthcare"]

data = get_data()

def get_payoff_matrices(data,election_outcome=False):
    payoffs = []
    for state in range(10):
        dem_ps = []
        rep_ps = []
        for dem_s in range(4):
            dem_ps.append([])
            rep_ps.append([])
            for rep_s in range(4):
                dem_p, rep_p = calculate_payoff(dem_s, rep_s, state, data, election_outcome=election_outcome)
                dem_ps[dem_s].append(dem_p)
                rep_ps[dem_s].append(rep_p)
        payoffs.append([dem_ps,rep_ps])
    return payoffs

# Function to calculate the Nash equilibrium

def calculate_nash_equilibrium(payoff_matrices):
    game = Game(payoff_matrices[0], payoff_matrices[1])
    equilibria = game.support_enumeration()
    return list(equilibria)


def find_dominant_strategy(payoff_matrices):
    dem_m=np.array(payoff_matrices[0])
    rep_m=np.array(payoff_matrices[1])
    rows=list(range(3))
    cols=list(range(3))
    no_next_move=False
    while not no_next_move and (len(rows)>1 or len(cols)>1):
        stop=False
        for i in rows:
            for ii in rows:
                if i!=ii and not stop:
                    row1=dem_m[i,cols]
                    row2=dem_m[ii,cols]
                    if np.all(row1>row2):
                        rows.remove(ii)
                        stop=True
        if not stop:
            for i in cols:
                for ii in cols:
                    if i!=ii and not stop:
                        col1=rep_m[rows,i]
                        col2=rep_m[rows,ii]
                        if np.all(col1>col2):
                            cols.remove(ii)
                            stop=True
        if not stop:
            no_next_move=True
    if len(rows)==1 and len(cols)==1:
        return rows[0],cols[0]
    else:
        return None

def find_weak_dominant_strategy(payoff_matrices):
    dem_m=np.array(payoff_matrices[0])
    rep_m=np.array(payoff_matrices[1])
    rows=list(range(3))
    cols=list(range(3))
    no_next_move=False
    while not no_next_move and (len(rows)>1 or len(cols)>1):
        stop=False
        for i in rows:
            for ii in rows:
                if i!=ii and not stop:
                    row1=dem_m[i,cols]
                    row2=dem_m[ii,cols]
                    if np.all(row1>=row2):
                        rows.remove(ii)
                        stop=True
        if not stop:
            for i in cols:
                for ii in cols:
                    if i!=ii and not stop:
                        col1=rep_m[rows,i]
                        col2=rep_m[rows,ii]
                        if np.all(col1>=col2):
                            cols.remove(ii)
                            stop=True
        if not stop:
            no_next_move=True
    if len(rows)==1 and len(cols)==1:
        return rows[0],cols[0]
    else:
        return None


def output_nash_results(f,equilibria,payoffs):
    f.write('\n')
    for equi_i,equi in enumerate(equilibria):
        f.write('Nash Equilibrium '+str(equi_i+1)+': \n')
        
        dem_s=np.argmax(equi[0]==1)
        rep_s=np.argmax(equi[1]==1)
        #f.write('Democrats: \n')
        #for i in range(4):
        #    f.write(str(equi[0][i])+' X '+dem_strat[i]+'\n')
        #f.write('Republicans: \n')
        #for i in range(4):
        #    f.write(str(equi[1][i])+' X '+rep_strat[i]+'\n')

        f.write('Democrates should do: '+dem_strat[dem_s]+'\n')
        f.write('Republicans should do: '+rep_strat[rep_s]+'\n')
        
        f.write('Dem results: '+str(payoffs[0][dem_s][rep_s])+'\n')
        f.write('Rep results: '+str(payoffs[1][dem_s][rep_s])+'\n')
        f.write('\n')
    f.write('\n')

def output_dom_results(f,res,election_outcome,payoffs):
    f.write('Democrates should do: '+dem_strat[res[0]]+'\n')
    f.write('Republicans should do: '+rep_strat[res[1]]+'\n')
    f.write('Dem results: '+str(payoffs[0][res[0]][res[1]])+'\n')
    f.write('Rep results: '+str(payoffs[1][res[0]][res[1]])+'\n')

def output_results(election_outcome=False):
    with open('output_election_outcome_'+str(election_outcome)+'.txt', 'w') as f:
        state_payoffs=get_payoff_matrices(data,election_outcome=election_outcome)
        for state,payoffs in enumerate(state_payoffs):
            f.write('Results for the state '+data['s_name'][state]+' ---------------------------------------------------\n\n')
            dom_strat=find_dominant_strategy(payoffs)
            if dom_strat is None:
                f.write('There are no strict dominant strategies\n')
                weak_dom_strat=find_weak_dominant_strategy(payoffs)
                if weak_dom_strat is None:
                    f.write('There are no weak dominant startegies\n')
                else:
                    f.write('There is a weak dominant strategy:\n')
                    output_dom_results(f,weak_dom_strat,election_outcome,payoffs)
            else:
                f.write('There is a strong dominant strategy:\n')
                output_dom_results(f,dom_strat,election_outcome,payoffs)
            equilibria=calculate_nash_equilibrium(payoffs)
            f.write('\nNash Equilibria:\n')
            output_nash_results(f,equilibria,payoffs)
        f.write('\n\n***********************\n\nRaw Results\n\n')
        for state,payoffs in enumerate(state_payoffs):
            f.write('\n\nResults for the state '+data['s_name'][state]+' :\n\n')
            f.write(str(payoffs))
            f.write('\n\n')

output_results()
output_results(election_outcome=True)

                    

