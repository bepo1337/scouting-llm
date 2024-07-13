from typing import List

def get_average(float_list: List[float]) -> float:
    return sum(float_list) / len(float_list)

def print_player_counts(player_counts: [float]):
    print("\n----- Player counts -----")
    print("Player count list: ", player_counts)
    print("Player counts len: ", len(player_counts))
    print(f"average percentage of players in model answer: {get_average(player_counts)}")