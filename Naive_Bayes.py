import csv

def build_naive_bayes_model(reader):
    '''This function builds the naive bayes model based on the first 20 messages contained in the dataset.

    :param reader: CSV reader that is reading the dataset
    :return: ham_dict - (key:value) of (word:# occurrences of word) for ham messages
    :return: spam_dict - (key:value) of (word:# occurrences of word) for spam messages'''

    # Initialize ham_dict and spam_dict. total_words tracks total number of words for each class of message, count tracks total number of each class occurence
    ham_dict = {'total_words': 0, 'count': 0}
    spam_dict = {'total_words': 0, 'count': 0}
    # Initialize unique_words to be used for laplace smoothing. Will be added to each dictionary total_words
    unique_words = 0
    # Counter used so that only the first 20 valid messages will be trained on
    counter = 0
    # Iterate through each row in the .csv file. Read the training data (first 20 messages in the csv file)
    for row in reader:
        # The first row details the format, contains no training data
        if row[0] == 'Target':
            continue
        # If the message is a ham message, increment ham_dict count
        elif row[0] == 'ham':
            ham_dict['count'] += 1
            # For each word in the message, increment its value in ham_dict if it already exists, or add it to ham_dict and set it to 1 and increment unique_words
            for word in row[1].split():
                word = word.lower()
                ham_dict['total_words'] += 1
                if word in ham_dict:
                    ham_dict[word] += 1
                else:
                    ham_dict[word] = 1
                    if word not in spam_dict:
                        unique_words += 1
        # If the message is a spam message, increment spam_dict count  
        elif row[0] == 'spam':
            spam_dict['count'] += 1
            # For each word in the message, increment its value in spam_dict if it already exists, or add it to spam_dict and set it to 1 and increment unique_words
            for word in row[1].split():
                word = word.lower()
                spam_dict['total_words'] += 1
                if word in spam_dict:
                    spam_dict[word] += 1
                else:
                    spam_dict[word] = 1
                    if word not in ham_dict:
                        unique_words += 1
        counter += 1
        if counter == 20:
            break

    # Apply laplace smoothing (add unique_words to total_words, 1 will be added to the numerator when conditional probability is calculated)
    ham_dict['total_words'] += unique_words
    spam_dict['total_words'] += unique_words

    return ham_dict, spam_dict

def test_naive_bayes_model(reader, ham_dict, spam_dict):
    '''This function tests the naive bayes model on the last 20 messages.
    
    :param reader: CSV reader that is reading the dataset
    :param ham_dict: Ham dictionary built using naive bayes
    :param spam_dict: Spam dictionary built using naive bayes
    :return test_results: List containing the test results for each test message'''

    test_results = [] # format [(actual_class, sentence, ham_prob, spam_prob, prediction, bool_correct), ...]
    # Calculate the prior probability of ham and spam. Will be used in posterior probability calculation
    prior_prob_ham, prior_prob_spam = compute_prior_probability(ham_dict, spam_dict)
    # Read the test data (last 10 messages in the csv file) and evaluate each message
    for row in reader:
        actual_class = row[0]
        sentence = row[1]
        # ham_probability and spam_probability will be used as rolling probabilities for each message to calculate posterior probability of each message
        ham_probability = 1
        spam_probability = 1
        # Evaluate each word in the message
        for word in row[1].split():
            word = word.lower()
            # Calculate the conditional probability of each word in the message for spam and ham
            p_ham, p_spam = compute_conditional_probability(word, ham_dict, spam_dict)
            # Maintain rolling posterior probability for the message
            ham_probability *= p_ham
            spam_probability *= p_spam
        # Complete posterior probability for the message by multiplying by the prior probabilities
        ham_probability *= prior_prob_ham
        spam_probability *= prior_prob_spam
        # Predict ham or spam depending on which probability is higher
        prediction = 'ham' if ham_probability > spam_probability else 'spam'
        # Add the results of each individual test to test_results
        test_results.append((actual_class, sentence, ham_probability, spam_probability, prediction, actual_class == prediction))
    return test_results

def compute_prior_probability(ham_dict, spam_dict):
    '''Computes the prior probability of ham or spam based on the naive bayes trained models.
    
    :param ham_dict: Ham dictionary built using naive bayes
    :param spam_dict: Spam dictionary built using naive bayes
    :return prior_prob_ham: Prior probability that a message was ham
    :return prior_prob_spam: Prior probability that a message was spam'''

    prior_prob_ham = ham_dict['count'] / (ham_dict['count'] + spam_dict['count'])
    prior_prob_spam = spam_dict['count'] / (ham_dict['count'] + spam_dict['count'])
    return prior_prob_ham, prior_prob_spam

def compute_conditional_probability(word, ham_dict, spam_dict):
    '''Computes the conditional probability of a word given that the message is spam or ham.

    :param word: Word found in testing message to be evaluated
    :param ham_dict: Ham dictionary built using naive bayes
    :param spam_dict: Spam dictionary built using naive bayes
    :return cond_prob_ham: Conditional probability that a word exists given that the message is ham
    :return cond_prob_spam: Conditional probability that a word ecists given that the message is spam'''
    # Determine number of times word occurs in ham messages. Apply laplace smoothing by adding 1 to the numerator, regardless of count
    if word in ham_dict:
        ham_numerator = ham_dict[word] + 1
    else:
        ham_numerator = 1
    # Determine number of times word occurs in spam messages. Apply laplace smoothing by adding 1 to the numerator, regardless of count
    if word in spam_dict:
        spam_numerator = spam_dict[word] + 1
    else:
        spam_numerator = 1
        
    cond_prob_ham = ham_numerator / ham_dict['total_words']
    cond_prob_spam = spam_numerator / spam_dict['total_words']

    return cond_prob_ham, cond_prob_spam
    

if __name__ == '__main__':
    csv_file = 'SpamDetection.csv'
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        ham_dict, spam_dict = build_naive_bayes_model(reader)
        test_results = test_naive_bayes_model(reader, ham_dict, spam_dict)
    f.close()
    for test in test_results:
        print(f'Actual Class: {test[0]}')
        print(f'Sentence: {test[1]}')
        print(f'Posterior probability of ham: {test[2]}')
        print(f'Posterior probability of spam: {test[3]}')
        print(f'Prediction: {test[4]}')
        print(f'{"Correct" if test[5] else "Incorrect"}\n')
