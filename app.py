import streamlit as st
from src.streamlit_tools.streamlit_game import GameStreamlit
from src.action_management import get_possible_actions
import math


# ---- Gama variables ----
player_number = 3
max_dice = 2
total_dice = player_number * max_dice

# ---- Initialize game only once ----
if "game" not in st.session_state:
    st.session_state.game = GameStreamlit(player_number=player_number,
                                   max_dice=max_dice)
    st.session_state.player_order = st.session_state.game.active_players.copy()

game = st.session_state.game
player_order = st.session_state.player_order


# ---- Gama info ----
st.header('Game data')
st.markdown(f"Player order: {list(player_order)}")

# -- Turn info --
st.header('Turn info')

n_dice, last_bet, turn_player, turn_player_hand = game.get_turn_info()

st.write(f"Turn player: {turn_player}\n")
st.write(f"Turn player hand: {turn_player_hand}\n")
st.write(f"Last bet: {last_bet}")

# ---- Action selection ----
action_list = get_possible_actions(last_bet=last_bet,
                                   total_dice=total_dice)

buttons_per_row = 8
n_rows = math.ceil(len(action_list) / buttons_per_row)

for row in range(n_rows):
    row_actions = action_list[row*buttons_per_row:(row+1)*buttons_per_row]
    cols = st.columns(buttons_per_row)

    for i in range(len(row_actions)):
        action = row_actions[i]
        if cols[i].button(str(action), key=f"{row}_{i}", use_container_width=True):
            outcome, message = st.session_state.game.make_a_bet(action)
            st.rerun()
            st.write(message)


# ---- Reset ----
if st.button("Reset game"):
    st.session_state.game = GameStreamlit(player_number=player_number,
                                   max_dice=max_dice)
    st.rerun()