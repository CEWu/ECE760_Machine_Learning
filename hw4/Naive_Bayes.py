import os
from glob import glob
from collections import Counter
import math

def read_file_to_wordlist(filepath):
    wordlists = []
    file_paths = [y for x in os.walk(filepath) for y in glob(os.path.join(x[0], '*.txt'))]
    for file_path in file_paths:
        with open(file_path) as f:
            # wordlist = f.read().splitlines()
            # wordlists.extend(wordlist)
            while(1):
                char = f.read(1)
                if char == '\n':
                    continue
                if not char:
                    break
                wordlists.append(char)
    return wordlists


def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    none_count = 0
    file_wordlist = []
    with open(filepath) as f:
        # file_wordlist = f.read().splitlines()
        while(1):
            char = f.read(1)
            if char == '\n':
                continue
            if not char:
                break
            file_wordlist.append(char)

    for vocab_word in vocab:
        vocab_count = 0
        for file_word in file_wordlist:
            if vocab_word == file_word:
                if bow.get(vocab_word) == None:
                    vocab_count += 1
                    bow[vocab_word] = vocab_count
                else:
                    bow[vocab_word] += 1

    for file_word in file_wordlist:
        if file_word not in vocab:
            none_count += 1
    if none_count >=1:
        bow[None] = none_count
    return bow

def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    dataset = []
    file_paths = [y for x in os.walk(directory) for y in glob(os.path.join(x[0], '*.txt'))]
    for file_path in file_paths:
        training_sample = {}
        label = file_path.split('/')[-2]
        bag_of_words = create_bow(vocab, file_path)
        training_sample['label'] = label
        training_sample['bow'] = bag_of_words
        dataset.append(training_sample)

    return dataset

def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    vocab = []
    wordlists = read_file_to_wordlist(directory)
    word_counter = Counter(wordlists).most_common()
    for word, count in word_counter:
        if count >= cutoff:
            vocab.append(word)

    vocab.sort()
    return vocab

def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    count_eng = 0
    count_jpn = 0
    count_spn = 0
    smooth = 0.5 # smoothing factor
    logprob = {}
    for training_sample in training_data:
        if training_sample['label'] == 'eng':
            count_eng += 1
        elif training_sample['label'] == 'jpn':
            count_jpn += 1
        elif training_sample['label'] == 'spn':
            count_spn += 1

    smooth_total_file = len(training_data) + 3 * smooth
    smooth_count_eng =  count_eng + smooth
    smooth_count_jpn =  count_jpn + smooth
    smooth_count_spn =  count_spn + smooth

    for label in label_list:
        if label == 'eng':
            # logprob[label] = smooth_count_eng / smooth_total_file
            logprob[label] = math.log10(smooth_count_eng / smooth_total_file)
        elif label == 'jpn':
            # logprob[label] = smooth_count_jpn / smooth_total_file
            logprob[label] = math.log10(smooth_count_jpn / smooth_total_file)
        elif label == 'spn':
            # logprob[label] = smooth_count_spn / smooth_total_file
            logprob[label] = math.log10(smooth_count_spn / smooth_total_file)
    return logprob

def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 0.5 # smoothing factor
    word_prob = {}
    all_word_counter = Counter()
    for training_sample in training_data:
        if training_sample['label'] == label:
            all_word_counter.update(training_sample['bow'])

    vocab_length = len(vocab)
    smooth_vocab_length = vocab_length * 0.5
    sum_of_words_count = sum(all_word_counter.values())
    smooth_sum_of_words_count = sum_of_words_count + smooth

    for vocab_word in vocab:
        word_count = all_word_counter[vocab_word]
        smooth_word_count = word_count + smooth
        # word_prob[vocab_word] = smooth_word_count / (smooth_sum_of_words_count + smooth * smooth_vocab_length)
        word_prob[vocab_word] = math.log10(smooth_word_count / (smooth_sum_of_words_count + smooth_vocab_length))

    if all_word_counter.get(None):
        none_word_count = all_word_counter[None]
        smooth_none_word_count = none_word_count + smooth
        # word_prob[None] = smooth_none_word_count / (smooth_sum_of_words_count + smooth * smooth_vocab_length)
        word_prob[None] = math.log10(smooth_none_word_count / (smooth_sum_of_words_count + smooth_vocab_length))
    else:
        #  word_prob[None] = (0 + smooth) / (smooth_sum_of_words_count + smooth * smooth_vocab_length)
        word_prob[None] = math.log10((0 + smooth) / (smooth_sum_of_words_count + smooth_vocab_length))
    return word_prob


##################################################################################
def train(training_directory, cutoff):
    retval = {}
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    log_prior = prior(training_data, ['eng', 'jpn', 'spn'])
    log_likelihood_jpn = p_word_given_label(vocab, training_data, 'jpn')
    log_likelihood_eng = p_word_given_label(vocab, training_data, 'eng')
    log_likelihood_spn = p_word_given_label(vocab, training_data, 'spn')

    retval['vocabulary'] = vocab
    retval['log prior'] = log_prior
    retval['log p(w|y=eng)'] = log_likelihood_eng
    retval['log p(w|y=jpn)'] = log_likelihood_jpn
    retval['log p(w|y=spn)'] = log_likelihood_spn


    return retval


def classify(model, filepath):
    retval = {}
    log_words_likelihood_eng = 0
    log_words_likelihood_jpn = 0
    log_words_likelihood_spn = 0
    test_file_wordlist = []
    # test_file_wordlist = read_file_to_wordlist(filepath)
    # with open(filepath) as f:
    #     test_file_wordlist = f.read().splitlines()
    with open(filepath) as f:
        # wordlist = f.read().splitlines()
        # wordlists.extend(wordlist)
        while(1):
            char = f.read(1)
            if char == '\n':
                continue
            if not char:
                break
            test_file_wordlist.append(char)
    log_prior_eng = model['log prior']['eng']
    log_prior_jpn = model['log prior']['jpn']
    log_prior_spn = model['log prior']['spn']
    for test_word in test_file_wordlist:
        if model['log p(w|y=eng)'].get(test_word) != None:
            log_word_likelihood_eng = model['log p(w|y=eng)'].get(test_word)
            log_words_likelihood_eng = log_words_likelihood_eng + log_word_likelihood_eng
        else:
            log_word_likelihood_eng = model['log p(w|y=eng)'].get(None)
            log_words_likelihood_eng = log_words_likelihood_eng + log_word_likelihood_eng

        if model['log p(w|y=jpn)'].get(test_word) != None:
            log_word_likelihood_jpn = model['log p(w|y=jpn)'].get(test_word)
            log_words_likelihood_jpn = log_words_likelihood_jpn + log_word_likelihood_jpn
        else:
            log_word_likelihood_jpn = model['log p(w|y=jpn)'].get(None)
            log_words_likelihood_jpn = log_words_likelihood_jpn + log_word_likelihood_jpn

        if model['log p(w|y=spn)'].get(test_word) != None:
            log_word_likelihood_spn = model['log p(w|y=spn)'].get(test_word)
            log_words_likelihood_spn = log_words_likelihood_spn + log_word_likelihood_spn
        else:
            log_word_likelihood_spn = model['log p(w|y=spn)'].get(None)
            log_words_likelihood_spn = log_words_likelihood_spn + log_word_likelihood_spn

    margin = log_words_likelihood_eng + log_words_likelihood_jpn + log_words_likelihood_spn
    log_posterior_eng = (log_prior_eng + log_words_likelihood_eng)/margin
    log_posterior_jpn = (log_prior_jpn + log_words_likelihood_jpn)/margin
    log_posterior_spn = (log_prior_spn + log_words_likelihood_spn)/margin

    result = [log_posterior_eng, log_posterior_jpn, log_posterior_spn]
    max_idx = result.index(max(result))
    if max_idx == 0:
        prediction = 'eng'
    elif max_idx == 1:
        prediction = 'jpn'
    elif max_idx == 2:
        prediction = 'spn'
    retval['predicted y'] = prediction
    retval['log p(y=eng|x)'] = log_words_likelihood_eng
    retval['log p(y=jpn|x)'] = log_words_likelihood_jpn
    retval['log p(y=spn|x)'] = log_words_likelihood_spn

    retval['log p(x|y=eng)'] = log_posterior_eng
    retval['log p(x|y=jpn)'] = log_posterior_jpn
    retval['log p(x|y=spn)'] = log_posterior_spn

    return retval


if __name__ == "__main__":
    # print(read_file_to_wordlist('./languageID/train/'))
    vocab  = create_vocabulary('./languageID/train/eng', 1)
    training_data = load_training_data(vocab, './languageID/train')
    # print('-----Q2-----')
    # print('eng')
    # # print(p_word_given_label(vocab, training_data, 'eng'))
    # tmp_eng = p_word_given_label(vocab, training_data, 'eng')
    # for char in tmp_eng:
    #     print('{}:{:.4f}'.format(char,10**(tmp_eng[char])),end = ', ')
    # print('')
    # print('-----Q3-----')
    # print('jpn')
    # tmp_jpn = p_word_given_label(vocab, training_data, 'jpn')
    # for char in tmp_jpn:
    #     print('{}:{:.4f}'.format(char,10**(tmp_jpn[char])),end = ', ')
    # print('')
    # print('spn')
    # tmp_spn = p_word_given_label(vocab, training_data, 'spn')
    # for char in tmp_spn:
    #     print('{}:{:.4f}'.format(char,10**(tmp_spn[char])),end = ', ')
    # print('')
    # print('-----Q4-----')
    # print(create_bow(vocab, './languageID/test/eng/e10.txt'))
    print('-----Q5-----')
    print('-----Q6-----')
    model = train('./languageID/train/', 1)
    classify_results = classify(model, './languageID/test/eng/e10.txt')
    print(classify_results)
    for key in classify_results:
        if key == 'predicted y':
            continue
        print('{}:{:.4f}'.format(key, 10**(classify_results[key])))
    # print('-----Q7-----')
    # test_file_paths = [y for x in os.walk('./languageID/test/') for y in glob(os.path.join(x[0], '*.txt'))]
    # for test_file_path in test_file_paths:
    #     print(test_file_path.split('/')[-1])
    #     print(classify(model, test_file_path)['predicted y'])
