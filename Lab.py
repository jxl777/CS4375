import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data_url = 'https://raw.githubusercontent.com/jxl777/CS4375/main/data1.csv'
data = pd.read_csv(data_url)

# Extract numerical values from columns with both numbers and text
data['TotalKill'] = data['TotalKill(Avg.)'].apply(lambda x: float(x.split()[0]))
data['TotalAssistant'] = data['TotalAssistant(Avg.)'].apply(lambda x: float(x.split()[0]))
data['TotalDeath'] = data['TotalDeath(Avg.)'].apply(lambda x: float(x.split()[0]))

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['Rank', 'KDA', 'MVP', 'TotalKill', 'TotalAssistant', 'TotalDeath', 
                      'AverageGoldGain', 'AverageFarming', 'AverageWarding', 
                      'AverageAntiWarding', 'AverageTeamfight%']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Extract team names correctly from the Player column
def extract_team(player_name):
    team = ""
    for i in range(len(player_name)):
        if player_name[i].isupper():
            team += player_name[i]
        else:
            break
    return team

data['Team'] = data['Player'].apply(extract_team)

# Group players by team
teams = data.groupby('Team')['Player'].apply(list).to_dict()

# Remove team prefix from player names
for team, players in teams.items():
    teams[team] = [player[len(team):] for player in players]

# Sort teams and players for consistent output
for team_name in sorted(teams.keys()):
    print(f"{team_name}:")
    for player in sorted(teams[team_name]):
        print(player) 
    print() # Add a blank line for separation
