from NaiveBayesClassifier import NaiveBayesClassifier as NB
from Preprocessing import Preprocessing
import os


def classify_test_dataset():
    # Functia de clasificare a fiecarui fisier, pe categorii, din folder-ul test_data
    try:
        for category in categories:
            current_folder = os.path.join(train_data_folder, category)
            files = os.listdir(current_folder)
            classifier = NB(spam_dict, ham_dict, categories)
            # spam_emails_number_category, ham_emails_number_category sunt numerele de email-uri ham si spam, pe categorii
            spam_emails_number_category, ham_emails_number_category = classifier.get_spam_and_ham_emails_by_category(
                files)

            for file in files:
                file_path = os.path.join(current_folder, file)
                classifier.classify(file_path, spam_emails_number_category, ham_emails_number_category)

    except IOError as exception:
        print('IO problem occurred:', exception)


email_folder_path = 'lingspam_public'
categories = ['bare', 'lemm', 'lemm_stop', 'stop']
train_data_folder = os.path.join(email_folder_path, 'train_data')
test_data_folder = os.path.join(email_folder_path, 'test_data')

preprocessing = Preprocessing(categories)
preprocessing.data_split(email_folder_path, train_data_folder, test_data_folder)
ham_dict, spam_dict = preprocessing.train_labels(train_data_folder)
filtered_ham_words = preprocessing.filter_top_words(ham_dict)
filtered_spam_words = preprocessing.filter_top_words(spam_dict)

print("Ham data:", filtered_ham_words)
print("Spam data:", filtered_spam_words)

classify_test_dataset()
