import nltk
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', download_dir='.')
nltk.download('wordnet', download_dir='/home/emilia/anaconda3/envs/nlp-a1/nltk_data')


def print_dict(dictionary):
    i = 0
    for key in dictionary:
        print('- ' + key + ' | ' + str(dictionary[key]))
        i += 1
        if i == 20:
            break


def print_type_token_ratio(dictionary):
    type = len(dictionary)
    token = 0
    for key in dictionary:
        token += dictionary[key]
    ratio = type / token * 100
    print('Type/token ratio: ' + str(ratio) + '%')


if __name__ == '__main__':
    wnl = WordNetLemmatizer()

    corpus = ''
    for filename in os.listdir('./CUAD_v1/full_contract_txt'):
        path = './CUAD_v1/full_contract_txt/' + filename
        with open(path) as file:
            corpus += file.read() + '\n'

    # Submit a file output.txt with the tokenizer’s output for the whole corpus. Include in your report  the first 20
    # lines from output.txt.
    tokens = word_tokenize(corpus)
    for i in range(len(tokens)):
        tokens[i] = wnl.lemmatize(tokens[i].lower())
    with open('./output.txt', 'w') as file:
        file.write('\n'.join(token for token in tokens))

    print('First 20 tokens:')
    for i in range(20):
        print('- ' + tokens[i])

    # How many tokens did you find in the corpus? How many types (unique tokens) did you have? What is the type/token
    # ratio for the corpus? The type/token ratio is defined as the number of types divided by the number of tokens.
    print('Number of tokens:' + str(len(tokens)))

    # For each token, print the token and its frequency in a file called tokens.txt (from the most frequent to the least
    # frequent) and include the first 20 lines in your report.
    frequency = {}
    for token in tokens:
        if token not in frequency:
            frequency[token] = 1
        else:
            frequency[token] += 1
    sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    frequency = {}
    for tup in sorted_frequency:
        frequency[tup[0]] = tup[1]

    print_type_token_ratio(frequency)

    with open('./tokens.txt', 'w') as file:
        for key in frequency:
            file.write(key + ' | ' + str(frequency[key]) + '\n')

    print('20 most frequent tokens:')
    print_dict(frequency)

    # How many tokens appeared only once in the corpus?
    once = 0
    for key in frequency:
        if frequency[key] == 1:
            once += 1
    print('Number of tokens only appearing once: ' + str(once))

    # From the list of tokens, extract only words, by excluding punctuation and other symbols, if any. Please pay
    # attention to end of sentence dot (full stops). How many words did you find? List the top 20 most frequent words in
    # your report, with their frequencies. What is the type/token ratio when you use only words (called lexical
    # diversity)?
    new_frequency = {}
    for key in frequency:
        if re.fullmatch(r'[a-z]+-?[a-z]*\.?', key) is not None:
            if re.fullmatch(r'.*\.', key):
                new_frequency[key[:-1]] = frequency[key]
            else:
                new_frequency[key] = frequency[key]

    print('Number of words: ' + str(len(new_frequency)))
    print('20 most frequent words:')
    print_dict(new_frequency)
    print_type_token_ratio(new_frequency)

    # From the list of words, exclude stopwords. List the top 20 most frequent words and their frequencies in your
    # report. You can use this list of stopwords (or any other that you consider adequate). Also compute the type/token
    # ratio when you use only word tokens without stopwords (called lexical density)?
    with open('./StopWords.txt') as file:
        stop_words = file.read().splitlines()

    new_new_frequency = {}
    for key in new_frequency:
        if key not in stop_words:
            new_new_frequency[key] = new_frequency[key]

    print('20 most frequent words without stop words:')
    print_dict(new_new_frequency)
    print_type_token_ratio(new_new_frequency)

    # Compute all the pairs of two consecutive words (bigrams) (excluding stopwords and punctuation). List the most
    # frequent 20 pairs and their frequencies in your report.
    new_tokens = []
    for token in tokens:
        if re.fullmatch(r'[a-z]+-?[a-z]*\.?', token) is not None:
            if re.fullmatch(r'.*\.', token) and token[:-1] not in stop_words:
                new_tokens.append(token[:-1])
            elif token not in stop_words:
                new_tokens.append(token)

    bigrams = {}
    for i in range(len(new_tokens)-1):
        bigram = new_tokens[i] + ', ' + new_tokens[i+1]
        if bigram not in bigrams:
            bigrams[bigram] = 1
        else:
            bigrams[bigram] += 1
    sorted_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
    bigrams = {}
    for tup in sorted_bigrams:
        bigrams[tup[0]] = tup[1]

    print('20 most frequent bigrams:')
    print_dict(bigrams)
