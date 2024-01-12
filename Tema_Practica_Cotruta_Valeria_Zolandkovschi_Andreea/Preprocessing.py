import os
import shutil
import re
from collections import Counter


class Preprocessing:
    def __init__(self, categories):
        self.categories = categories
        self.spam_emails = list()
        self.ham_emails = list()

    def data_split(self, folder_path, train_data_folder, test_data_folder):
        train_exists = os.path.exists(os.path.join(train_data_folder))
        test_exists = os.path.exists(os.path.join(test_data_folder))

        if not train_exists or not test_exists:
            for category in self.categories:
                for i in range(1, 11):
                    current_folder = os.path.join(folder_path, category, f'part{i}')
                    print("Current folder processing:", current_folder)
                    email_list = os.listdir(current_folder)

                    spam_emails = [file for file in email_list if 'spm' in file]
                    ham_emails = [file for file in email_list if 'spm' not in file]

                    if i == 10:
                        test_folder = os.path.join(test_data_folder, category)
                        os.makedirs(test_folder, exist_ok=True)
                        for file in spam_emails:
                            shutil.copy(os.path.join(current_folder, file), test_folder)
                        for file in ham_emails:
                            shutil.copy(os.path.join(current_folder, file), test_folder)
                    else:
                        train_folder = os.path.join(train_data_folder, category)
                        os.makedirs(train_folder, exist_ok=True)
                        for file in spam_emails:
                            shutil.copy(os.path.join(current_folder, file), train_folder)
                        for file in ham_emails:
                            shutil.copy(os.path.join(current_folder, file), train_folder)

    @staticmethod
    def extract_words(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def train_labels(self, train_data_folder):
        word_dictionary_ham = {}
        word_dictionary_spam = {}

        for category in self.categories:
            category_folder = os.path.join(train_data_folder, category)
            files = os.listdir(category_folder)

            for file in files:
                file_path = os.path.join(category_folder, file)

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()[1:]
                    text = ''.join(lines)

                    words = self.extract_words(text)
                    words = [word for word in words if word.isalpha()]

                    is_spam = 'spm' in file

                    if is_spam:
                        if category not in word_dictionary_spam:
                            word_dictionary_spam[category] = Counter()
                        word_dictionary_spam[category] += Counter(words)
                    else:
                        if category not in word_dictionary_ham:
                            word_dictionary_ham[category] = Counter()
                        word_dictionary_ham[category] += Counter(words)

        return word_dictionary_ham, word_dictionary_spam
