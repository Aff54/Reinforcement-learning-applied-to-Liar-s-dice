import streamlit as st

from src.game import GameRL


# ---- Game initialisation ----

game = GameRL(player_number = 3, max_dice = 2)

st.write(f"Turn player hand: {[hand.tolist() for hand in game.player_hands]}\n")


st.markdown(f"Player order: {list(game.active_players)}")

# Getting turn info.

n_dice, last_bet, turn_player, turn_player_hand = game.get_turn_info()

st.write(f"Turn player: {turn_player}\n")
st.write(f"Turn player hand: {turn_player_hand}\n")
st.write(f"Last bet: {last_bet}")

if st.button('Resetting game'):
    game = GameRL(player_number = 3, max_dice = 2)