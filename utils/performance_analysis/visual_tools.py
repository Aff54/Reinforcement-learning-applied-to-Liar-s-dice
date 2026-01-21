# ---- Regular Packages ----
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Functions
# --------------------------------------------------------

def plot_pie_charts(ranking_array):
    """Display the rate at which each player appeared at each place in 
    ranking_array as pie charts.

    Args:
        ranking_array (np.array): Array with i-row being the ranking of players 
        for the i-th game.
    """

    n_simulation, player_number = ranking_array.shape
    # Mapping players with colors => each player will always have the same color.
    colors = plt.cm.tab10(np.arange(player_number))  # 10 distinct colors
    player_colors = {f"player {i+1}": colors[i] for i in range(player_number)}
    
    # -- Plotting --
    fig, ax = plt.subplots(1, player_number, figsize=(4 * player_number, 4))

    # Plotting place representation pie chart in sequence.
    for place in range(1, player_number + 1):
        labels = ["player " + str(i+1) 
                  for i in range(player_number) 
                  if sum(ranking_array[:, place - 1] == i+1) > 1]
        
        sizes = [sum(ranking_array[:, place-1] == i+1)/n_simulation 
                 for i in range(player_number) 
                 if sum(ranking_array[:, place -1] == i+1) > 1]
    
        pie_colors = [player_colors[label] 
                      for label in labels]
    
        wedges, texts, autotexts = ax[place - 1].pie(sizes, 
                                                     labels=labels, 
                                                     autopct='%1.1f%%', 
                                                     colors = pie_colors)
        for autotext in autotexts:
            autotext.set_color('white')
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(place, "th")
        _ = ax[place - 1].set_title(f"{str(place) + suffix} place distribution among players")

    fig.tight_layout()

    return fig