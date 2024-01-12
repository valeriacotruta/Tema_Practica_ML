import os
import matplotlib.pyplot as plt
from Preprocessing import Preprocessing
from NaiveBayesClassifier import NaiveBayesClassifier as NB


class CrossValidationLeaveOneOut:
    def __init__(self, spam_dict, ham_dict, categories):
        self.spam_dict = spam_dict
        self.ham_dict = ham_dict
        self.categories = categories

    def remove_from_dictionary(self, dictionary, category_index, words):
        for word in words:
            word_counter = dictionary.get(self.categories[category_index])[word]
            dictionary.get(self.categories[category_index])[word] = word_counter - 1

    def remove_by_category(self, file_path, words, is_spam=True):
        if is_spam:
            dictionary = self.spam_dict
        else:
            dictionary = self.ham_dict
        if self.categories[0] in file_path:
            self.remove_from_dictionary(dictionary, 0, words)
        elif self.categories[1] in file_path and '_stop' not in file_path:
            self.remove_from_dictionary(dictionary, 1, words)
        elif self.categories[2] in file_path:
            self.remove_from_dictionary(dictionary, 2, words)
        elif self.categories[3] in file_path and '_stop' not in file_path:
            self.remove_from_dictionary(dictionary, 3, words)

    def leave_one_out_cross_validation(self, directory_path):
        accuracies = []
        for root, dirs, files in os.walk(directory_path):
            for file_name in files:
                if file_name.endswith('.txt'):
                    file_path = os.path.join(root, file_name)

                    is_spam = 'spm' in file_path

                    file = open(file_path, 'r')
                    lines = file.readlines()[1:]
                    text = ''.join(lines)

                    words = Preprocessing.extract_words(text)
                    words = [word for word in words if word.isalpha()]
                    spam_emails, ham_emails = 0, 0
                    if is_spam:
                        spam_emails -= 1
                        self.remove_by_category(file_path, words)
                    else:
                        ham_emails -= 1
                        self.remove_by_category(file_path, words, is_spam)

                    classifier = NB(self.spam_dict, self.ham_dict, self.categories)
                    spam_emails_number_category, ham_emails_number_category = classifier.get_spam_and_ham_emails_by_category(
                        files)
                    spam_emails_number_category += spam_emails
                    ham_emails_number_category += ham_emails
                    current_email_label = classifier.classify(file_path, spam_emails_number_category,
                                                              ham_emails_number_category, words)
                    accuracies.append(1 if (current_email_label == 'spam' and is_spam) or (
                            current_email_label == 'ham' and not is_spam) else 0)

                self.__init__(self.spam_dict, self.ham_dict,
                              self.categories)

        accuracy_percentage = sum(accuracies) / len(accuracies) * 100
        return accuracy_percentage

    @staticmethod
    def plot_cross_validation_results(results):
        plt.figure(figsize=(8, 6))
        plt.hist(results, bins=10, alpha=0.7, color='blue')
        plt.xlabel('Accuratețe (%)')
        plt.ylabel('Frecvență')
        plt.title('Distribuția Accurateții Leave-One-Out Cross-Validation dupa 10 rulari.')
        plt.grid(True)
        plt.show()
