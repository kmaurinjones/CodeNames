import streamlit as st
import random
import time
import pandas as pd
import math

import tensorflow as tf
import tensorflow_hub as hub
# Load the Universal Sentence Encoder's TF Hub module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def initialize_board(rows, columns):
    # Ensure no duplicates on the board
    board = []
    while len(board) < rows * columns:
        word = random.choice(vocab)
        if word not in board:
            board.append(word)
    return [board[i:i + columns] for i in range(0, rows * columns, columns)]

def use_ai_guess(hint, board, number):
    hint_embed = embed([hint])
    word_scores = []

    for word in sum(board, []):
        word_embed = embed([word])
        score = tf.keras.losses.cosine_similarity(hint_embed, word_embed)
        word_scores.append((word, -score.numpy()[0]))  # Negating score because cosine similarity returns values between -1 and 1, where 1 means completely similar

    # Sort by highest similarity score
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, score in word_scores[:number]]

def find_closest_factors(target):
    # Calculate the square root of the target number
    sqrt_target = math.sqrt(target)

    # Find the integer closest to the square root
    closest_int = round(sqrt_target)

    # Check if the square of this integer is the target number
    if closest_int * closest_int == target:
        return closest_int, closest_int

    # If not, search for the closest factor pair
    lower_int = closest_int
    upper_int = closest_int + 1

    while lower_int > 0:
        if target % lower_int == 0:
            upper_int = target // lower_int
            break
        lower_int -= 1

    return lower_int, upper_int

def find_nearly_square_factors(n):
    """ 
    Finds numbers up to n that can be divided into a rectangle with nearly equal dimensions.
    A 'nearly square' factorization is defined here as one where the ratio of the larger dimension
    to the smaller one is less than or equal to 2.
    """
    nearly_square_numbers = []

    for i in range(1, n + 1):
        for j in range(1, int(i ** 0.5) + 1):
            if i % j == 0:  # j is a factor of i
                other_factor = i // j
                # Check if the dimensions are nearly equal (ratio <= 2)
                if max(j, other_factor) / min(j, other_factor) <= 2:
                    nearly_square_numbers.append(i)
                    break  # No need to check other factors for this number

    return nearly_square_numbers

# Load in full vocabulary (adjust path as necessary)
vocab = [line.strip() for line in open("data/vocab.txt", "r").readlines() if line.strip()]

def reset_game(rows, columns):
    st.session_state['board_state'] = initialize_board(rows, columns)
    st.session_state['user_selected_cards'] = set()
    st.session_state['game_summary'] = []
    st.session_state['turn'] = 1
    st.session_state['turn_messages'] = []
    st.session_state['user_hint'] = ""
    st.session_state['guesses_correct'] = []
    st.session_state['guesses_total'] = []
    st.session_state['turn_skill'] = []

def ai_turn():
    st.session_state['turn'] += 1
    turn = st.session_state['turn']
    number = len(st.session_state['user_selected_cards'])
    st.write("Spybot is thinking...")
    st.session_state['turn_messages'].append("Spybot is thinking...")
    guesses = use_ai_guess(st.session_state['user_hint'], st.session_state['board_state'], number)
    correct_guesses = 0
    for idx, guess in enumerate(guesses):
        guess_text = f"> Spybot's guess: *{guess.upper()}*"
        st.session_state['turn_messages'].append(guess_text)
        
        time.sleep(1)  # Delay for dramatic effect
        
        st.write(guess_text)
        if guess in st.session_state['user_selected_cards']:
            correct_guesses += 1
            st.session_state['board_state'] = [[word for word in row if word != guess] for row in st.session_state['board_state']]
            st.write(f"> --- Guess is *correct*! {idx+1}/{len(guesses)}")
        else:
            st.write(
                f"> --- Guess is *incorrect*. {idx}/{len(guesses)} \
                correct this turn. \
                {sum(len(row) for row in st.session_state['board_state'])} \
                cards remaining ---"
                )
            break

    # printout about player skill this game
    st.session_state['guesses_correct'].append(correct_guesses)
    st.session_state['guesses_total'].append(len(guesses))
    skill_val = round(sum(st.session_state['guesses_correct']) / sum(st.session_state['guesses_total']), 2) # all correct / all incorrect -> will be higher than avg % correct
    st.session_state['turn_skill'].append(skill_val) # add it to skill vals log

    # base message printed at each turn
    message = f"Spybot's review of your skill: {round(skill_val * 100)}%."

    # line for spacing
    st.write("-----------------------------")
    if turn-1 > 1:
        prev_turn_skill = st.session_state['turn_skill'][-2]
        curr_turn_skill = st.session_state['turn_skill'][-1]
        
        # if improving
        if curr_turn_skill > prev_turn_skill:
            skill_message = " You're improving!"
        elif curr_turn_skill < prev_turn_skill:
            skill_message = " You're slipping!"
        else: # if it's unchanged since last turn
            skill_message = " You're consistent!"

        # add skill message suffix to normal message if game is over
        if all(len(row) == 0 for row in st.session_state['board_state']):
            if skill_val <= 0.5:
                skill_message = " Well done. You've finished the board. Think of ways you can improve for next time."
            elif 0.5 < skill_val <= 0.7:
                skill_message = " You're doing well enough, but there's still room for improvement."
            elif 0.7 < skill_val <= 0.85:
                skill_message = " You're doing great! You're on track to being a pro."
            elif 0.85 < skill_val:
                skill_message = " You're either a professional or a cheater. You should be proud or ashamed!"

        message += skill_message

    # write skill eval message
    st.write(message)

    st.session_state['game_summary'].append({
        "Turn": turn-1, # to account for =+ from earlier
        "Selected Cards": ", ".join(st.session_state['user_selected_cards']).upper(),
        "Spybot Correct Guesses": correct_guesses,
        "Used Hints": str(st.session_state['user_hint']),
        "Cards Remaining": sum(len(row) for row in st.session_state['board_state'])
    })
    st.session_state['user_selected_cards'] = set()
    st.session_state['user_hint'] = ""
    st.session_state['turn_messages'] = []

def toggle_card_selection(card):
    if card in st.session_state['user_selected_cards']:
        st.session_state['user_selected_cards'].remove(card)
    else:
        st.session_state['user_selected_cards'].add(card)

# Streamlit interface
if __name__ == "__main__":
    app_name = "Codenames: Agent AI"
    st.title(app_name)
    
    # line for spacing
    st.write("-----------------------------")

    # game instructions
    st.write("**Game instructions:**")
    st.write("Ever played Codenames? Spybot is your personal guessing buddy \
             and you have to work together to get through the whole \
             board in as few turns as possible!")
    st.write("1. At the start of each turn, select the cards that relate to your hint")
    st.write("2. Type your hint into the box in the form")
    st.write('3. Click the "Submit Turn" button at the bottom of the form')
    st.write('4. Watch Spybot try to solve its way through your guesses!')
    st.write('5. Complete the board in as few turns as possible. You are your only competition. Good luck!')

    # line for spacing
    st.write("-----------------------------")

    st.write("How many cards would you like to use? We suggest \
            starting with 15 and increasing or decreasing the \
            amount depending on your desired difficulty.")


    # Slider for selecting the number of cards
    n = 100  # the most cards that can be on the board
    nearly_square_numbers = find_nearly_square_factors(n)

    num_cards = st.select_slider(
        'Select the number of cards for the board',
        options=nearly_square_numbers,
        value=15  # Default value
    )

    rows, columns = find_closest_factors(num_cards) # get x and y for board

    # "Initialize Game" button
    if 'game_initialized' not in st.session_state:
        st.session_state['game_initialized'] = False

    if not st.session_state['game_initialized']:
        if st.button("Initialize Game"):
            st.session_state['game_initialized'] = True
            rows, columns = find_closest_factors(num_cards)
            reset_game(rows, columns)

    if st.session_state['game_initialized']:

        # Reset game button
        if st.button("Reset Game"):
            st.session_state['game_initialized'] = False
            reset_game(rows, columns)

        # User input for the hint
        with st.form("user_input"):
            # Display selected cards within the form
            st.write(f"**Turn #{st.session_state['turn']}**")
            st.write("Selected cards:")
            if len(st.session_state["user_selected_cards"]) > 0:
                st.write(f'*{", ".join(card.upper() for card in st.session_state["user_selected_cards"])}*')

            # Get user hint
            user_hint = st.text_input("Enter your hint", value=st.session_state['user_hint']).lower()

            # Check if the user hint or any part of it is in the board words
            hint_invalid = any(word in [word for row in st.session_state['board_state'] for word in row] for word in user_hint.split())

            # Display a warning message if the hint is invalid
            if hint_invalid:
                st.warning("No part of your hint can exist on the board. Please enter another hint.")

            # Only allow form submission if the hint is valid and at least one card is selected
            if not hint_invalid:
                # submit_input = st.form_submit_button("Submit Turn", clear_on_submit = True) # this doesn't seem to work despite being in docs. Current version too old?
                submit_input = st.form_submit_button("Submit Turn")

                if submit_input and len(st.session_state["user_selected_cards"]) < 1:
                    st.warning("You must select at least one card for Spybot to guess!")

        # Display messages for the current turn
        for message in st.session_state['turn_messages']:
            st.write(message)

        if submit_input:
            if len(st.session_state["user_selected_cards"]) >= 1:
                st.session_state['user_hint'] = user_hint  # Update the session state with the latest hint
                ai_turn()
                # Clearing text field after turn is submitted
                st.session_state['user_hint'] = ""

        # line for spacing
        st.write("-----------------------------")
        
        # Check if all cards have been guessed
        if all(len(row) == 0 for row in st.session_state['board_state']):
            game_over_text = f"Congratutions! You've successfully prompted Spybot to guess \
                all {num_cards} cards in {st.session_state['turn']} turns. Great work! \
                Click the button below to download a log of your game."
            
            st.write(game_over_text)

            # File format selection for download
            file_format = st.selectbox('Select File Format for Download:', ['CSV', 'JSON', 'HTML'])

            # Download button
            # if st.button('Download Game Summary'):
            summary_df = pd.DataFrame(st.session_state['game_summary'])
            if file_format == 'CSV':
                st.download_button(label="Download CSV", data=summary_df.to_csv(index=False), file_name="game_summary.csv", mime="text/csv")
            elif file_format == 'JSON':
                st.download_button(label="Download JSON", data=summary_df.to_json(orient='records'), file_name="game_summary.json", mime="application/json")
            elif file_format == 'HTML':
                st.download_button(label="Download HTML", data=summary_df.to_html(index=False), file_name="game_summary.html", mime="text/html")

        # Display and interact with the game board
        for row in range(rows):
            cols = st.columns(columns)
            for col in range(columns):
                with cols[col]:
                    # Check if the row exists in the board state and if the column index is within the range of the row
                    if row < len(st.session_state['board_state']) and col < len(st.session_state['board_state'][row]):
                        word = st.session_state['board_state'][row][col]
                        # Create a checkbox for each word
                        is_selected = st.checkbox(word.upper(), key=f"card-{row}-{col}")
                        if is_selected:
                            st.session_state['user_selected_cards'].add(word)
                        elif word in st.session_state['user_selected_cards']:
                            st.session_state['user_selected_cards'].remove(word)

        # line for spacing
        st.write("-----------------------------")

        # Display the game summary
        if st.session_state['game_summary']:
            st.markdown("### GAME SUMMARY:")
            summary_df = pd.DataFrame(st.session_state['game_summary'])
            st.table(summary_df)

# Disclaimer + Author Note
st.write("**Disclaimer**")
st.write(
    "*This game is in no way affiliated with the official Codenames game, \
    Vlaada Chvátil, or Czech Games Edition. This is simply a fun web app \
    I made to explore an aspect of quantitative linguistics about which I am passionate. \
    This is not monetized, nor will it be at any time in the future.*"
    )

st.write(
    f"Thanks for checking out {app_name}! If you have any feedback or requests \
    for additions to this app, shoot me an email at kmaurinjones@gmail.com."
    )