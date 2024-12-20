import os
from sklearn.linear_model import LogisticRegression
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import requests
import streamlit as st

data = os.path.abspath("nltk_data")
nltk.data.path.append(data)
nltk.download('punkt', download_dir=data)

#intents
INTENTS = [
    {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"], "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]},
    {"tag": "goodbye", "patterns": ["Bye", "See you later", "Goodbye", "Take care"], "responses": ["Goodbye", "See you later", "Take care"]},
    {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"], "responses": ["You're welcome", "No problem", "Glad I could help"]},
    {"tag": "about", "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"], "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]},
    {"tag": "help", "patterns": ["Help", "I need help", "Can you help me", "What should I do"], "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]},
    {"tag": "age", "patterns": ["How old are you", "What's your age"], "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]},
    {"tag": "weather", "patterns": ["What's the weather like", "How's the weather today"], "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]},
    {"tag": "budget", "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"], "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]},
    {"tag": "credit_score", "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"], "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]},
    {"tag": "travel", "patterns": ["How can I book a flight", "What are the best travel destinations", "Where should I travel next"], "responses": ["You can book a flight using popular websites like Expedia, Google Flights, or directly on airline websites.", "Some of the best travel destinations include Paris, Tokyo, and Bali.", "Consider your interests when choosing a destination. Do you prefer beaches, mountains, or historical sites?"]},
    {"tag": "fitness", "patterns": ["How can I lose weight", "What are the best exercises", "How do I stay fit"], "responses": ["To lose weight, focus on a balanced diet and regular exercise.", "Some of the best exercises include running, swimming, and strength training.", "Staying fit requires consistency and a mix of cardio and strength exercises."]},
    {"tag": "technology", "patterns": ["What is AI", "Explain machine learning", "What are the latest tech trends"], "responses": ["AI stands for Artificial Intelligence, which involves machines simulating human intelligence.", "Machine learning is a subset of AI where machines learn from data.", "Some of the latest tech trends include AI advancements, 5G technology, and blockchain."]},
    {"tag": "health", "patterns": ["How can I stay healthy", "What are the benefits of drinking water", "What is a balanced diet"], "responses": ["Staying healthy involves regular exercise, a balanced diet, and adequate sleep.", "Drinking water helps maintain bodily functions and improves skin health.", "A balanced diet includes fruits, vegetables, proteins, and healthy fats."]},
    {"tag": "education", "patterns": ["What are the best ways to study", "How do I prepare for exams", "What are good study techniques"], "responses": ["Some of the best ways to study include active recall and spaced repetition.", "Prepare for exams by reviewing notes, practicing past papers, and staying organized.", "Good study techniques include creating summaries, using flashcards, and taking breaks."]},
    {"tag": "sports", "patterns": ["What are the rules of soccer", "How do I start playing basketball", "What is the best sport to stay active"], "responses": ["Soccer involves two teams trying to score goals by getting the ball into the opposing team's net.", "To start playing basketball, learn the basic rules and practice dribbling and shooting.", "The best sport to stay active depends on your interests. Popular options include swimming, running, and tennis."]},
    {"tag": "movies", "patterns": ["What are the best movies to watch", "Can you recommend a good movie", "What is your favorite movie"], "responses": ["Some of the best movies to watch include 'The Shawshank Redemption', 'Inception', and 'The Godfather'.", "Try watching a classic like 'Forrest Gump' or a recent hit like 'Dune'.", "I don't watch movies, but I can suggest some based on popular opinions!"]},
    {"tag": "books", "patterns": ["What are the best books to read", "Can you recommend a book", "What is your favorite book"], "responses": ["Some great books include '1984' by George Orwell, 'To Kill a Mockingbird' by Harper Lee, and 'The Great Gatsby' by F. Scott Fitzgerald.", "If you like fantasy, try 'The Lord of the Rings'. If you prefer mystery, 'Gone Girl' is a great pick.", "I don't read books, but I can suggest some based on popular opinions!"]},
    {"tag": "cooking", "patterns": ["How do I bake a cake", "What are easy dinner recipes", "How do I cook pasta"], "responses": ["To bake a cake, mix flour, sugar, eggs, and baking powder, then bake in a preheated oven.", "Some easy dinner recipes include stir-fry, pasta, and tacos.", "To cook pasta, boil water, add pasta, and cook until al dente."]},
    {"tag": "career", "patterns": ["How do I write a resume", "What are good interview tips", "How do I choose a career path"], "responses": ["To write a resume, focus on your skills, experience, and achievements.", "Good interview tips include researching the company, practicing common questions, and dressing appropriately.", "Choosing a career path involves assessing your interests, strengths, and goals."]},
    {"tag": "coding", "patterns": ["How do I learn Python", "What are the best coding languages", "How can I debug my code"], "responses": ["You can learn Python through online courses like Codecademy or free resources like freeCodeCamp.", "Popular coding languages include Python, JavaScript, and C++.", "To debug your code, use tools like debuggers or try printing intermediate values."]},
    {"tag": "finance", "patterns": ["What is compound interest", "How do I invest in stocks", "What is cryptocurrency"], "responses": ["Compound interest is when interest earns on both the initial principal and the accumulated interest.", "You can invest in stocks through brokerages like Robinhood or Fidelity.", "Cryptocurrency is a digital or virtual currency secured by cryptography."]},
    {"tag": "food", "patterns": ["What is the best dessert", "How do I make a smoothie", "What are healthy snacks"], "responses": ["The best dessert depends on your taste, but chocolate cake is a classic choice.", "To make a smoothie, blend your favorite fruits like bananas, berries, and spinach with a liquid such as almond milk or yogurt.", "Healthy snacks include nuts, fruits, veggies with hummus, or Greek yogurt with honey."]}


]



# Initialize
vectorizer = TfidfVectorizer()
logReg = LogisticRegression(random_state=0, max_iter=10000)

# Chatbot response
def get_response(input):
    vec = vectorizer.transform([input])
    predict_tag = logReg.predict(vec)[0]
    for intent in INTENTS:
        if intent['tag'] == predict_tag:
            return random.choice(intent['responses'])

# Preprocess data
patterns, tag = [], []
for intent in INTENTS:
    for pat in intent['patterns']:
        tok = word_tokenize(pat)
        new_pattern = " ".join(tok)
        patterns.append(new_pattern)
        tag.append(intent['tag'])

# Training
x_train = vectorizer.fit_transform(patterns)
y_train = tag
logReg.fit(x_train, y_train)



# Streamlit
def main():
    st.title("Chatbot")
    st.write("Write a message in the box below to start a conversation with the chatbot!")
    inp = st.text_input("You:", key="user_input")

    if inp:
        response = get_response(inp)
        st.text_area("Chatbot Response:", value=response, height=75, key="response_output")

        if response.lower() in ["goodbye", "bye", "bye bye"]:
            st.write("Thank you for using this chatbot!")
            st.stop()


if __name__ == "__main__":
    main()
