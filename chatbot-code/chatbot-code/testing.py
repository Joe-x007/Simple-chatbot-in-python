# # ChatBot project
# import nltk, random, json , pickle
# #nltk.download('punkt');nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.feature_extraction.text import CountVectorizer

# lemmatizer = WordNetLemmatizer()
# context = {}

# class Testing:
#     def __init__(self):
#         # Load the intent file
#         self.intents = json.loads(open('intents.json').read())
#         # Load the training_data file which contains training data
#         data = pickle.load(open("training_data", "rb"))
#         self.words = data['words']
#         self.classes = data['classes']
#         self.model = load_model('chatbot_model.h5')
#         # Set the error threshold value
#         self.ERROR_THRESHOLD = 0.5
#         self.ignore_words = list("!@#$%^&*?")

#     def clean_up_sentence(self, sentence):
#         # Tokenize each sentence (user's query)
#         sentence_words = word_tokenize(sentence.lower())
#         # Lemmatize the word to root word and filter symbol words
#         sentence_words = list(map(lemmatizer.lemmatize, sentence_words))
#         sentence_words = list(filter(lambda x: x not in self.ignore_words, sentence_words))
#         return set(sentence_words)

#     def wordvector(self, sentence):
#         # Initialize CountVectorizer
#         # txt.split helps to tokenize single character
#         cv = CountVectorizer(tokenizer=lambda txt: txt.split())
#         sentence_words = ' '.join(self.clean_up_sentence(sentence))
#         words = ' '.join(self.words)

#         # Fit the words into cv and transform into one-hot encoded vector
#         vectorize = cv.fit([words])
#         word_vector = vectorize.transform([sentence_words]).toarray().tolist()[0]
#         return np.array(word_vector) 

#     def classify(self, sentence):
#         # Predict to which class(tag) user's query belongs
#         results = self.model.predict(np.array([self.wordvector(sentence)]))[0]
#         # Store the class name and probability of that class 
#         results = list(map(lambda x: [x[0], x[1]], enumerate(results)))
#         # Accept those class probabilities greater than threshold value (0.5)
#         results = list(filter(lambda x: x[1] > self.ERROR_THRESHOLD, results))
#         # Sort class probability values in descending order
#         results.sort(key=lambda x: x[1], reverse=True)
#         return_list = []

#         for i in results:
#             return_list.append((self.classes[i[0]], str(i[1])))
#         return return_list
    
#     def results(self, sentence, userID):
#         # If context is maintained then filter class(tag) accordingly
#         if sentence.isdecimal():
#             if context.get(userID) == "historydetails":
#                 return self.classify('ordernumber')
#         return self.classify(sentence)
    
#     def response(self, sentence, userID='TechVidvan'):
#         # Get class of user's query
#         results = self.results(sentence, userID)
#         print(sentence, results)
#         # Store random response to the query
#         ans = ""
#         if results:
#             while results:
#                 for i in self.intents['intents']:
#                     # Check if tag == query's class
#                     if i['tag'] == results[0][0]:
#                         # If class contains key as "set"
#                         # then store key as userID along with its value in
#                         # context dictionary
#                         if 'set' in i and 'filter' not in i:
#                             context[userID] = i['set']
#                         # If the tag doesn't have any filter, return response
#                         if 'filter' not in i:
#                             ans = random.choice(i['responses'])
#                             print("Query:", sentence)
#                             print("Bot:", ans)
#                         # If a class has key as filter, then check if context dictionary key's value is same as filter value
#                         # Return the random response
#                         if userID in context and 'filter' in i and i['filter'] == context[userID]:
#                             if 'set' in i:
#                                 context[userID] = i['set']
#                             ans = random.choice(i['responses'])
#                 results.pop(0)
#         # If ans contains some value then return response to user's query else return some message
#         return ans if ans != "" else "Sorry! I am still learning.\nYou can train me by providing more data."

# ChatBot project
import nltk, random, json, pickle
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = WordNetLemmatizer()
context = {}

class Testing:
    def __init__(self):
        # Get absolute path of the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'intents.json')
        training_data_path = os.path.join(current_dir, 'training_data')
        model_path = os.path.join(current_dir, 'chatbot_model.h5')
        
        # Load the intent file with error handling
        try:
            with open(file_path, 'r') as file:
                self.intents = json.load(file)
        except FileNotFoundError:
            print(f"Error: intents.json not found at {file_path}")
            exit(1)
        
        # Load the training_data file which contains training data
        try:
            with open(training_data_path, "rb") as file:
                data = pickle.load(file)
        except FileNotFoundError:
            print(f"Error: training_data not found at {training_data_path}")
            exit(1)
        
        self.words = data['words']
        self.classes = data['classes']
        
        # Load the model with error handling
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            exit(1)
        
        # Set the error threshold value
        self.ERROR_THRESHOLD = 0.5
        self.ignore_words = list("!@#$%^&*?")

    def clean_up_sentence(self, sentence):
        # Tokenize each sentence (user's query)
        sentence_words = word_tokenize(sentence.lower())
        # Lemmatize the word to root word and filter symbol words
        sentence_words = list(map(lemmatizer.lemmatize, sentence_words))
        sentence_words = list(filter(lambda x: x not in self.ignore_words, sentence_words))
        return set(sentence_words)

    def wordvector(self, sentence):
        # Initialize CountVectorizer
        cv = CountVectorizer(tokenizer=lambda txt: txt.split())
        sentence_words = ' '.join(self.clean_up_sentence(sentence))
        words = ' '.join(self.words)

        # Fit the words into cv and transform into one-hot encoded vector
        vectorize = cv.fit([words])
        word_vector = vectorize.transform([sentence_words]).toarray().tolist()[0]
        return np.array(word_vector)

    def classify(self, sentence):
        # Predict to which class(tag) user's query belongs
        results = self.model.predict(np.array([self.wordvector(sentence)]))[0]
        # Store the class name and probability of that class 
        results = list(map(lambda x: [x[0], x[1]], enumerate(results)))
        # Accept those class probabilities greater than threshold value (0.5)
        results = list(filter(lambda x: x[1] > self.ERROR_THRESHOLD, results))
        # Sort class probability values in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []

        for i in results:
            return_list.append((self.classes[i[0]], str(i[1])))
        return return_list
    
    def results(self, sentence, userID):
        # If context is maintained then filter class(tag) accordingly
        if sentence.isdecimal():
            if context.get(userID) == "historydetails":
                return self.classify('ordernumber')
        return self.classify(sentence)
    
    def response(self, sentence, userID='TechVidvan'):
        # Get class of user's query
        results = self.results(sentence, userID)
        print(sentence, results)
        # Store random response to the query
        ans = ""
        if results:
            while results:
                for i in self.intents['intents']:
                    # Check if tag == query's class
                    if i['tag'] == results[0][0]:
                        # If class contains key as "set"
                        # then store key as userID along with its value in
                        # context dictionary
                        if 'set' in i and 'filter' not in i:
                            context[userID] = i['set']
                        # If the tag doesn't have any filter, return response
                        if 'filter' not in i:
                            ans = random.choice(i['responses'])
                            print("Query:", sentence)
                            print("Bot:", ans)
                        # If a class has key as filter, then check if context dictionary key's value is same as filter value
                        # Return the random response
                        if userID in context and 'filter' in i and i['filter'] == context[userID]:
                            if 'set' in i:
                                context[userID] = i['set']
                            ans = random.choice(i['responses'])
                results.pop(0)
        # If ans contains some value then return response to user's query else return some message
        return ans if ans != "" else "Sorry! I am still learning.\nYou can train me by providing more data."
