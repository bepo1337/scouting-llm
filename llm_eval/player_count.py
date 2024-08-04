from typing import List

def get_average(float_list: List[float]) -> float:
    return sum(float_list) / len(float_list)

def print_player_counts(player_counts: [float], results_filename):
    print("\n----- Player counts -----")
    with open(results_filename, "a") as file:
        print("Player count list (should all be <=1): ", player_counts)
        file.write(f"Player count list (should all be <=1): {player_counts}\n")
        print("Player counts len: ", len(player_counts))
        file.write(f"Player counts len: {len(player_counts)}\n")
        print(f"average percentage of players in model answer: {get_average(player_counts)}")
        file.write(f"average percentage of players in model answer: {get_average(player_counts)}\n")
