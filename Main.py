import math
import operator
import csv

# Input: Dictionary which is emission_probability[tag]
# Output: Number of words which appears once in the given tag.
# This function is for good turing smoothing algorithm.
def find_words_appears_once(temp_dict):
    count = 0
    for key, value in temp_dict.items():
        if (value == 1):
            count += 1
    return count

#Input: Two list which are train sentences and test sentences
#Output: Number of difference unique words which is unkown words count.
def find_unknown_words_number(list1, list2):
    temp_set1 = set()
    temp_set2 = set()

    for sentence in list1:
        for word_tag_pair in sentence:
            word = word_tag_pair[0]
            temp_set1.add(word)

    for sentence in list2:
        for word_tag_pair in sentence:
            word = word_tag_pair[0]
            temp_set2.add(word)

    return len(temp_set2 - temp_set1)

# Input: Path of file
# Output: sentences of this given file. [[["word", "tag"], ["word", "tag"], ..]], [..]]]
# This function reads data from file and applies preprocessing.
def dataset(folderPath):
    train_file = open(folderPath, 'r')
    lines = train_file.readlines()

    sentences = list()
    sentence = list()
    sentence_pair = list()

    for line in lines:
        if (line == "\n"):      # Newline is a sentence boundary.
            if (len(sentence) == 0):    # If multiple newlines appears consecutively
                continue
            sentences.append(sentence)  # sentences = [[sentence], [sentence], ..]
            sentence = list()
        elif(line.strip().split()[0] == "-DOCSTART-"):
            continue
        else:                  # Create sentence
            if (line.strip().split()[3] == "B-ORG" or line.strip().split()[3] == "I-ORG"):
                sentence_pair.append(line.strip().split()[0].lower())
                sentence_pair.append("org")
                sentence.append(sentence_pair)  # sentence = [[sentence_pair], [sentence_pair]..]
                sentence_pair = list()          # sentence_pair = ["word", "tag"]

            elif (line.strip().split()[3] == "B-MISC" or line.strip().split()[3] == "I-MISC"):
                sentence_pair.append(line.strip().split()[0].lower())
                sentence_pair.append("misc")
                sentence.append(sentence_pair)
                sentence_pair = list()

            elif (line.strip().split()[3] == "B-PER" or line.strip().split()[3] == "I-PER"):
                sentence_pair.append(line.strip().split()[0].lower())
                sentence_pair.append("per")
                sentence.append(sentence_pair)
                sentence_pair = list()

            elif (line.strip().split()[3] == "B-LOC" or line.strip().split()[3] == "I-LOC"):
                sentence_pair.append(line.strip().split()[0].lower())
                sentence_pair.append("loc")
                sentence.append(sentence_pair)
                sentence_pair = list()

            else:
                sentence_pair.append(line.strip().split()[0].lower())
                sentence_pair.append("o")
                sentence.append(sentence_pair)
                sentence_pair = list()

    return sentences

# Input: Sentences which is in the training dataset. [[[word, tag], [word, tag], .., [word, tag]], .. , [[word, tag], [word, tag], .., [word, tag]]]
# Output: Hidden Markov Model
# This function takes training dataset sentences and calculates initial probability, transition probability and emission probabiality
def HMM(sentences, tags):

    # Initial frequency
    for sentence in sentences:
        for i in range(len(sentence)):
            tag = sentence[i][1]
            if (i == 0):
                initial_frequency[tag] = initial_frequency.get(tag, 0) + 1
                tags.add(tag)
            else:
                initial_frequency[tag] = initial_frequency.get(tag, 0) + 0
                tags.add(tag)

    # Initial probability
    initial_tag_count = sum(initial_frequency.values())
    for each_tag in tags:
        initial_probability[each_tag] = 0

    for key, value in initial_frequency.items():
        initial_probability[key] = value / initial_tag_count

    for tag in tags:
        transition_frequency[tag] = dict()
        transition_probability[tag] = dict()
        emission_frequency[tag] = dict()
        emission_probability[tag] = dict()

    # Transition Frequency
    tag_sentences = list()
    for sentence in sentences:
        tag_sentence = ""
        for token in sentence:
            tag_sentence += token[1] + " "
        tag_sentences.append(tag_sentence)

    # Generating bigram tag pairs
    all_pairs = list()
    for sentence in tag_sentences:
        tags = sentence.split()
        pairs = zip(*[tags[i:] for i in range(2)])
        all_pairs.append([" ".join(pair) for pair in pairs])

    for sentence_pairs in all_pairs:
        for pair in sentence_pairs:
            first_tag = pair.split(" ")[0]
            next_tag = pair.split(" ")[1]
            if (first_tag in transition_frequency.keys()):
                transition_frequency[first_tag].update({next_tag: transition_frequency[first_tag].get(next_tag, 0) + 1})

    # Transition Probability
    for tag, tags_dict in transition_frequency.items():
        total = 0
        for tag1, frequency1 in tags_dict.items():
            total += frequency1
        for tag2, frequency2 in tags_dict.items():
            transition_probability[tag][tag2] = frequency2 / total

    # Emission Frequency
    for sentence in sentences:
        for word_tag_pair in sentence:
            word = word_tag_pair[0]
            tag = word_tag_pair[1]
            if (tag in emission_frequency.keys()):
                emission_frequency[tag].update({word: emission_frequency[tag].get(word, 0) + 1})

    for tag in emission_frequency.keys():
        words_appears_once_tag_dict[tag] = find_words_appears_once(emission_frequency[tag])

    # Emission Probability
    for tag, tags_dict in emission_frequency.items():
        total = 0
        for tag1, frequency1 in tags_dict.items():
            total += frequency1
        for tag2, frequency2 in tags_dict.items():
            emission_probability[tag][tag2] = frequency2 / total
        # Smoothing part
        emission_probability[tag]["UNKNOWN"] = words_appears_once_tag_dict[tag] / (unknown_words_count * total)

# Input current word, possible tags for state, previous probabilities
# Output maximum state probability
# This function looks for every tag and calculate possibilities according to previous probabilities and find maximum probability.
def get_probability(word, tag, prev_probs):
    tag_probs = list()
    for prev_tag, prev_prob in prev_probs.items():
        try:            # If transition probability is zero, prevent log(0) error
            prob_transition = math.log(transition_probability[prev_tag].get(tag, 0), 2)
        except:
            prob_transition = 0
        prob_emission = math.log(emission_probability[tag].get(word, emission_probability[tag]["UNKNOWN"]), 2)
        prob_total = (prev_tag, prev_prob[1] + prob_transition + prob_emission)     # state probability
        tag_probs.append(prob_total)
    return max(tag_probs, key=operator.itemgetter(1))

# Input: sentence = [["word", "tag"], ["word", "tag"], ..]]
# Output: predicted tags for given sentence = ["tag", "tag", .. ]
# This function tries to predict name entity of words in the given sentence
def viterbi(sentence):
    predicted_tags = list()
    probabilities = []

    # For the first word
    first_word = sentence[0][0]
    init_probs = dict()
    for tag in tags:
        temp_tuple = ("-", math.log(initial_probability[tag], 2) + math.log(
            emission_probability[tag].get(first_word, emission_probability[tag]["UNKNOWN"]), 2))
        init_probs.update({tag: temp_tuple})
    probabilities.append(init_probs)

    # Rest of the words in the sentence
    for word_tag_pair in sentence[1:]:
        word = word_tag_pair[0]
        real_tag = word_tag_pair[1]
        temp_probs = dict()
        for tag in tags:
            p = get_probability(word, tag, probabilities[-1])
            if (p[1] == 0):
                continue
            temp_probs[tag] = p
        probabilities.append(temp_probs)
    probabilities.reverse()

    # Find maximum probability for backtracing
    max_value = probabilities[0].get(list(probabilities[0].keys())[0])[1]
    for key, value in probabilities[0].items():
        if(value[1] > max_value):
            max_value = value[1]

    # final tag and previous tag
    for key,value in probabilities[0].items():
        if(value[1] == max_value):
            tag = key
            prev_tag = value[0]

    # final tag
    predicted_tags.append(tag)

    # backtracing
    for prob in probabilities[1:]:
        predicted_tags.append(prev_tag)
        prev_tag = prob[prev_tag][0]
    predicted_tags.reverse()

    for tag in predicted_tags:
        predicted_tags_csv.append(tag)

    return predicted_tags

# Input: 1) predicted tags generated from viterbi algorithm. [["o", "misc", .. ], ["org", "o", .. ]] and test sentences
# Output: Number of correct predicted tags and number of total tags
# This function is for finding accuracy of model.
# There is one more evaluation function below this function.
# In the preprocessing I made "B-TAG" and "T-TAG" into "tag". For example I made "B-ORG" and "I-ORG" to "org"
# But for kaggle I had to change them to original positions.
# This function calculates accuracy according to preprocessed tags ("org", "o", "loc", "per", "misc")
# But other evaluation function which is csv_evaluation function calculates accuracy according to original tags ("B-ORG", "I-ORG",
# "B-LOC", "I-LOC", "B-PER", "I-PER", "B-MISC", "I-MISC", "O")
def evaluation(predicted_tags, test_sentences):
    correct_prediction = 0
    total_tag_count = 0

    correct_tags_sentences = list()
    for sentence in test_sentences:
        correct_tags_sentence = list()
        for word_tag_pair in sentence:
            correct_tags_sentence.append(word_tag_pair[1])
        correct_tags_sentences.append(correct_tags_sentence)

    for i, sentence in enumerate(test_sentences_list):
        predicted_tags_sentence = predicted_tags[i]
        for j, word_tag_pair in enumerate(sentence):
            tag = word_tag_pair[1]
            if (tag == predicted_tags_sentence[j]):
                correct_prediction += 1
            total_tag_count += 1

    return correct_prediction, total_tag_count

# Input1: Predicted tags of all sentence = [["o", "org", "o", "o"],["o", ... ]]
# Input2: Test dataset = [[["word", "tag"], ["word", "tag"] ..]], [["word", "tag"], ..]], ..]]]
# Output1: Correct predicted tag count
# Output2: Total tag count
def csv_evaluation(predictions, test_dataset):
    correct = 0
    total = 0

    # Change preprocessed tags into original ones for predicted tags. Firstly made all of them B-TAG except O
    for count, sentence in enumerate(predictions):
        for i in range(len(sentence)):
            if(sentence[i] == "o"):
                predictions[count][i] = "O"
            elif(sentence[i] == "org"):
                predictions[count][i] = "B-ORG"
            elif(sentence[i] == "misc"):
                predictions[count][i] = "B-MISC"
            elif(sentence[i] == "loc"):
                predictions[count][i] = "B-LOC"
            elif(sentence[i] == "per"):
                predictions[count][i] = "B-PER"

    # After look for consecutive same tags and make the next one I-TAG
    for count, sentence in enumerate(predictions):
        for i in range(len(sentence)-1):
            first_tag_index = i
            next_tag_index = i+1
            # Since there is no B-O or I-O, do not look for O tag
            if(sentence[first_tag_index][0] != "O" and sentence[next_tag_index][0] != "O"):
                splitted_first_tag = sentence[first_tag_index].split("-")[1]
                splitted_next_tag = sentence[next_tag_index].split("-")[1]
                if(splitted_first_tag == splitted_next_tag):
                    predictions[count][next_tag_index] = "I-" + sentence[next_tag_index].split("-")[1]

    # Change preprocessed tags into original ones for test data
    for count, sentence in enumerate(test_dataset):
        for i in range(len(sentence)):
            if(sentence[i][1] == "o"):
                test_dataset[count][i][1] = "O"
            elif(sentence[i][1] == "org"):
                test_dataset[count][i][1] = "B-ORG"
            elif(sentence[i][1] == "misc"):
                test_dataset[count][i][1] = "B-MISC"
            elif(sentence[i][1] == "loc"):
                test_dataset[count][i][1] = "B-LOC"
            elif(sentence[i][1] == "per"):
                test_dataset[count][i][1] = "B-PER"

    for count, sentence in enumerate(test_dataset):
        for i in range(len(sentence)-1):
            first_tag_index = i
            next_tag_index = i+1
            if(sentence[first_tag_index][1][0] != "O" and sentence[next_tag_index][1][0] != "O"):
                splitted_first_tag = sentence[first_tag_index][1].split("-")[1]
                splitted_next_tag = sentence[next_tag_index][1].split("-")[1]
                if(splitted_first_tag == splitted_next_tag):
                    test_dataset[count][next_tag_index][1] = "I-" + sentence[next_tag_index][1].split("-")[1]

    # write to csv file
    with open('b21426716-2ass.csv', mode='w') as output_file:
        fieldnames = ["Id", "Category"]
        output_writer = csv.writer(output_file, delimiter=',')
        output_writer.writerow(fieldnames)

        for i, sentence in enumerate(test_dataset):
            real_words = [pair[0] for pair in sentence]
            real_tags = [pair[1] for pair in sentence]
            predicted_tags1 = predictions[i]

            for i in range(len(predicted_tags1)):
                if predicted_tags1[i] == real_tags[i]:
                    correct += 1
                total += 1
                output_writer.writerow([total, predicted_tags1[i]])
    return correct, total


if __name__ == '__main__':

    train_data = "train.txt"
    test_data = "test.txt"

    train_sentences_list = dataset(train_data)
    test_sentences_list = dataset(test_data)

    initial_frequency = dict()
    initial_probability = dict()
    transition_frequency = dict()
    transition_probability = dict()
    emission_frequency = dict()
    emission_probability = dict()

    # unique tags
    tags = set()

    words_appears_once_tag_dict = dict()

    # Finding number of unknown words
    unknown_words_count = find_unknown_words_number(train_sentences_list, test_sentences_list)

    # Task 1: Create Hidden Markov Model
    HMM(train_sentences_list, tags)

    predicted_tags = list()

    # Task 2: Viterbi Algorithm
    # Predict test data using viterbi algorithm
    predicted_tags_csv = list()
    for sentence in test_sentences_list:
        predicted_tags.append(viterbi(sentence))

    # Task 3: Evaluation

    correct_tag_count = 0
    total_test_word_count = 0

    correct_tag_count, total_test_word_count = evaluation(predicted_tags, test_sentences_list)

    print("Correct predicted tag count: " + str(correct_tag_count))
    print("Total tag count: "+ str(total_test_word_count))
    print("Accuracy: " + str((correct_tag_count / total_test_word_count) * 100))

    # correct, total = csv_evaluation(predicted_tags, test_sentences_list)
    # print("Correct predicted tag count: " + str(correct))
    # print("Total tag count: "+ str(total))
    # print("Accuracy: " + str(correct * 100 / total))