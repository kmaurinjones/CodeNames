# Codenames: Agent AI

"Codenames: Agent AI" is a unique, single-player version of the popular Codenames game, enhanced with artificial intelligence. In this Streamlit web app, you, the player, provide hints and select cards to be guessed, while the AI, powered by Google's Universal Sentence Encoder, makes guesses based on the top-ranked similar cards.

Steamlit App link: [agentai.streamlit.app]

## Game Instructions

Ever played Codenames? Meet Spybot, your personal guessing buddy in "Codenames: Agent AI". Work together with Spybot to get through the entire board in as few turns as possible!

Here's how you play:

1. **Select Cards**: At the start of each turn, select the cards that relate to your hint.
2. **Type Your Hint**: Enter your hint into the provided box.
3. **Submit Turn**: Click the "Submit Turn" button at the bottom of the form.
4. **Spybot's Guess**: Watch as Spybot tries to solve its way through your guesses.
5. **Goal**: Complete the board in as few turns as possible. Challenge yourself and beat your own high score!

## Technologies Used

- **Streamlit**: For the web app interface.
- **TensorFlow & TensorFlow Hub**: Leveraging Google's Universal Sentence Encoder for generating embeddings and interpreting hints.
- **Pandas**: For data handling and manipulation.

## Setup and Installation

To run "Codenames: Agent AI" locally, follow these steps:

1. **Clone the Repository**:
git clone [https://github.com/kmaurinjones/CodeNames]

2. **Install Requirements**:
Navigate to the cloned repository and install the required packages using:
$ pip install -r requirements.txt

3. **Run the App**:
Inside the repository, start the Streamlit app by running:
$ streamlit run app.py

## License

MIT License (see LICENSE file)

## Contact

Kai Maurin-Jones - kmaurinjones@gmail.com
