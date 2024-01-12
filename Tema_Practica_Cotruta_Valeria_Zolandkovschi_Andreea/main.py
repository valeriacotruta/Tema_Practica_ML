from NaiveBayesClassifier import NaiveBayesClassifier as NB
from CrossValidationLeaveOneOut import CrossValidationLeaveOneOut as CVLOO
from Preprocessing import Preprocessing
import os
import matplotlib.pyplot as plt


def classify_test_dataset():
    try:
        correct = 0
        files_number = 0
        for category in categories:
            current_folder = os.path.join(test_data_folder, category)
            files = os.listdir(current_folder)
            classifier = NB(spam_dict, ham_dict, categories)
            spam_emails_number_category, ham_emails_number_category = classifier.get_spam_and_ham_emails_by_category(
                files)

            for file in files:
                file_path = os.path.join(current_folder, file)
                classification = classifier.classify(file_path, spam_emails_number_category, ham_emails_number_category)
                is_spam = 'spm' in file_path
                correct += (
                    1 if (classification == 'spam' and is_spam) or (classification == 'ham' and not is_spam) else 0)
                files_number += 1

        return correct / files_number

    except IOError as exception:
        print('IO problem occurred:', exception)


def print_naive_bayes_accuracy():
    print("Acuratetea algoritmului Bayes Naiv pe setul de date de testare:", accuracy_naive_bayes*100)

    plt.bar(['Acuratețe'], [accuracy_naive_bayes], color='blue')
    plt.ylim(0, 1)
    plt.ylabel('Acuratețe')
    plt.title('Performanța algoritmului Bayes Naiv pe setul de date de testare')
    plt.show()


def print_cvloo_accuracy():
    validation_results = []
    cvloo = CVLOO(spam_dict, ham_dict, categories)

    for _ in range(10):
        accuracy = cvloo.leave_one_out_cross_validation(train_data_folder)
        validation_results.append(accuracy)

    print("Distribuția Accurateții Leave-One-Out Cross-Validation dupa 10 rulari:", validation_results)
    print("Acuratetea algoritmului Leave-One-Out Cross-Validation pe setul de date de antrenare:",
          sum(validation_results) / len(validation_results))

    cvloo.plot_cross_validation_results(validation_results)


email_folder_path = 'lingspam_public'
categories = ['bare', 'lemm', 'lemm_stop', 'stop']
train_data_folder = os.path.join(email_folder_path, 'train_data')
test_data_folder = os.path.join(email_folder_path, 'test_data')

preprocessing = Preprocessing(categories)
preprocessing.data_split(email_folder_path, train_data_folder, test_data_folder)
ham_dict, spam_dict = preprocessing.train_labels(train_data_folder)

accuracy_naive_bayes = classify_test_dataset()
print_naive_bayes_accuracy()
print_cvloo_accuracy()
