import math

from Preprocessing import *


class NaiveBayesClassifier:
    # Clasificarea pe categorii
    def __init__(self, spam_emails_words_dictionary, ham_emails_words_dictionary, categories):
        self.spam_words_dictionary = spam_emails_words_dictionary
        self.ham_words_dictionary = ham_emails_words_dictionary
        self.categories = categories

    def classify(self, file_path, spam_emails_number_category, ham_emails_number_category):
        try:
            ham_probability = spam_emails_number_category / (
                    spam_emails_number_category + ham_emails_number_category)
            spam_probability = ham_emails_number_category / (
                    spam_emails_number_category + ham_emails_number_category)

            current_file = open(file_path, 'r')
            lines = current_file.readlines()[1:]
            text = ''.join(lines)
            words = Preprocessing.extract_words(text)
            words = [word for word in words if word.isalpha()]

            if self.categories[0] in file_path:
                spam_probability = self.compute_probability(words,
                                                            self.spam_words_dictionary.get(self.categories[0]),
                                                            spam_emails_number_category, spam_probability)
                ham_probability = self.compute_probability(words,
                                                           self.ham_words_dictionary.get(self.categories[0]),
                                                           ham_emails_number_category, ham_probability)
            elif self.categories[1] in file_path and '_stop' not in file_path:
                spam_probability = self.compute_probability(words,
                                                            self.spam_words_dictionary.get(self.categories[1]),
                                                            spam_emails_number_category, spam_probability)
                ham_probability = self.compute_probability(words,
                                                           self.ham_words_dictionary.get(self.categories[1]),
                                                           ham_emails_number_category, ham_probability)
            elif self.categories[2] in file_path:
                spam_probability = self.compute_probability(words,
                                                            self.spam_words_dictionary.get(self.categories[2]),
                                                            spam_emails_number_category, spam_probability)
                ham_probability = self.compute_probability(words,
                                                           self.ham_words_dictionary.get(self.categories[2]),
                                                           ham_emails_number_category, ham_probability)
            elif self.categories[3] in file_path and '_stop' not in file_path:
                spam_probability = self.compute_probability(words,
                                                            self.spam_words_dictionary.get(self.categories[3]),
                                                            spam_emails_number_category, spam_probability)
                ham_probability = self.compute_probability(words,
                                                           self.ham_words_dictionary.get(self.categories[3]),
                                                           ham_emails_number_category, ham_probability)
            print(f"file path: {file_path}, spam probability: {spam_probability}, ham probability: {ham_probability}, "
                  f"{'spam' if spam_probability > ham_probability else 'ham'}")

        except IOError:
            print('IO problem occurred: ' + file_path)

    @staticmethod
    def compute_probability(file_words, word_dictionary, total_emails_number, probability):
        vocabulary_size = len(word_dictionary)

        for word in file_words:
            if word in word_dictionary.keys():
                probability += math.log((word_dictionary[word] + 1) / (total_emails_number + vocabulary_size))
            else:
                probability += math.log(1 / (total_emails_number + vocabulary_size))

        return probability

    @staticmethod
    def get_spam_and_ham_emails_by_category(files):
        spam_emails = [file for file in files if 'spm' in file]
        total_spam_emails_number = len(spam_emails)
        ham_emails = [file for file in files if 'spm' not in file]
        total_ham_emails_number = len(ham_emails)
        return total_spam_emails_number, total_ham_emails_number
