import streamlit as st
from language_model import Chatbot

import speech_recognition as sr  # For speech-to-text
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Sentiment Analysis Class
class SentimentAnalyzer:
    def __init__(self):
        pass

    def analyze(self, text):
        # Preprocess and clean the text
        lower_case = text.lower()
        cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

        # Tokenize words
        tokenized_words = word_tokenize(cleaned_text, "english")

        # Remove stopwords
        final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

        # Lemmatize words
        lemma_words = [WordNetLemmatizer().lemmatize(word) for word in final_words]

        # Load emotion words and match with lemmatized words
        emotion_list = []
        with open('emotions.txt', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                word, emotion = clear_line.split(':')

                if word in lemma_words:
                    emotion_list.append(emotion)

        # Count emotions
        emotion_counts = Counter(emotion_list)

        # Analyze sentiment using VADER
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(cleaned_text)

        # Determine sentiment based on VADER scores
        if score['neg'] > score['pos']:
            sentiment = "Negative"
        elif score['neg'] < score['pos']:
            sentiment = "Positive"
        else:
            sentiment = "Neutral"

        # Return sentiment and emotion counts
        return sentiment, emotion_counts



st.title("AI Chatbot with Sentiment Analysis")


chatbot = Chatbot()
sentiment_analyzer = SentimentAnalyzer()
recognizer = sr.Recognizer()  

# Initialize session state for chat history and additional states
if "messages" not in st.session_state:
    st.session_state.messages = []  # List to store chat history

if "user_input" not in st.session_state:
    st.session_state.user_input = ""  # To store temporary user input

# Chat History Display
st.subheader("Chat History")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"You: {msg['content']}")
    else:
        st.write(f"Chatbot: {msg['content']}")

# User Input Section
st.subheader("Your Input")
user_input = st.text_input("Type your message here:", key="unique_user_input")

# Speech-to-Text Conversion
st.subheader("Or Speak Your Input")
uploaded_audio = st.file_uploader("Upload your audio file (in .wav format):", type=["wav"])

if uploaded_audio is not None:
    try:
        # Use the recognizer to process the uploaded audio
        with sr.AudioFile(uploaded_audio) as source:
            audio_data = recognizer.record(source)
            spoken_text = recognizer.recognize_google(audio_data)
            st.success(f"Recognized Speech: {spoken_text}")
            user_input = spoken_text  # Use the recognized text as input
    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {e}")

if st.button("Send"):
    if user_input:
        # Sentiment Analysis
        sentiment, emotion_counts = sentiment_analyzer.analyze(user_input)

        # Display Sentiment
        st.markdown(
            f"""
            <div style='padding:10px; border-radius:5px; background-color:#e9ecef; color:#343a40; font-size:18px; font-weight:bold;'>
            Sentiment: {sentiment} 
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display Emotion Counts (optional)
        if emotion_counts:
            st.markdown(
                """
                <div style='padding:10px; border-radius:5px; background-color:#f8f9fa; color:#495057; font-size:16px; font-weight:bold;'>
                Emotion Counts:
                </div>
                """,
                unsafe_allow_html=True,
            )
            for emotion, count in emotion_counts.items():
                st.write(f"{emotion}: {count}")

        # Append user input to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate Chatbot Response
        response = chatbot.generate_response(user_input)

        # Display Chatbot Response
        st.markdown(
            f"""
            <div style='padding:15px; border-radius:8px; background-color:#f8f9fa; color:#495057; 
            font-size:16px; font-weight:bold; border: 1px solid #dee2e6; margin-top:10px;'>
            🤖 Chatbot Response: {response}
            </div>
            """,
            unsafe_allow_html=True,
                    )
        st.session_state.messages.append({"role": "chatbot", "content": response})

        # Stop conversation if "bye" is entered
        if user_input.strip().lower() == "bye":
            st.write("Chatbot: Goodbye! 👋")
            st.stop()
            
