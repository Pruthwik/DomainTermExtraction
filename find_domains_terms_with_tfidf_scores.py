"""Create a baseline for term extraction using tfidf vectorizer."""
import os
from pickle import dump
from sys import argv
from nltk import ngrams
from nltk import word_tokenize
from collections import Counter
from math import log


def find_file_paths_from_folder(folder_path):
    """Find file paths from a folder."""
    for root, dirs, files in os.walk(folder_path):
        return [os.path.join(root, fl) for fl in files]


def read_lines_and_create_term_frequencies(file_paths, ngrm=3):
    """Read lines from files and create a generator of lines."""
    file_wise_term_freq_dict = dict()
    total_terms = set()
    for file_path in file_paths:
        file_name = file_path[file_path.rfind('/') + 1:]
        with open(file_path, 'r', encoding='utf-8') as file_read:
            term_freq_dict = Counter()
            for line in file_read:
                line = line.strip()
                words = word_tokenize(line)
                word_ngrams = list()
                for n in range(1, ngrm + 1):
                    n_grams = ngrams(words, n)
                    word_ngrams += [' '.join(grams) for grams in n_grams]
                term_freq_dict.update(word_ngrams)
            total_terms.update(term_freq_dict)
        file_wise_term_freq_dict[file_name] = term_freq_dict
    print(len(total_terms))
    return file_wise_term_freq_dict


def create_tfidf_dict_for_all_words(file_wise_term_freq_dict, min_df, max_df):
    """Create TFIDF dict for all words."""
    term_wise_tfidf_dict = dict()
    total_files = len(file_wise_term_freq_dict)
    for fl in file_wise_term_freq_dict:
        total_ngrm_freq = 0
        found_in_files = 0
        for ngrm in file_wise_term_freq_dict[fl]:
            if ngrm in term_wise_tfidf_dict:
                continue
            found_term_freq = file_wise_term_freq_dict[fl][ngrm]
            total_ngrm_freq += file_wise_term_freq_dict[fl][ngrm]
            found_in_files += 1
            for oth_fl in set(file_wise_term_freq_dict) - {fl}:
                if ngrm in file_wise_term_freq_dict[oth_fl]:
                    found_in_files += 1
                    total_ngrm_freq += file_wise_term_freq_dict[oth_fl][ngrm]
            if found_in_files >= min_df and found_in_files <= max_df:
                tf = found_term_freq / total_ngrm_freq
                idf = log(total_files / (1 + found_in_files))
                if ngrm not in term_wise_tfidf_dict:
                    term_wise_tfidf_dict[ngrm] = tf * idf
                    # if idf == log(total_files / 6):
                    #     print(ngrm)
    return term_wise_tfidf_dict


def write_lines_to_file(file_path, lines):
    """Read lines from a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def main():
    """Pass arguments and call functions here."""
    input_folder = argv[1]
    domain = argv[2]
    ngrm = 3
    min_df, max_df = 1, 5
    input_file_paths = find_file_paths_from_folder(input_folder)
    file_wise_term_freq_dict = read_lines_and_create_term_frequencies(input_file_paths, ngrm)
    term_wise_tfidf_dict = create_tfidf_dict_for_all_words(file_wise_term_freq_dict, min_df, max_df)
    print(len(term_wise_tfidf_dict))
    print(term_wise_tfidf_dict)
    write_lines_to_file(domain + 'tf-idf-max-df-' + str(max_df) + '-min-df-' + str(min_df) + '.txt', list(term_wise_tfidf_dict.keys()))
    # dump_object_to_pickle_file(domain + '-vectorizer-min-df-' + str(min_df) + '-max-df-' + str(max_df) + '-word-' + '-'.join(map(str, ngram_range)) + '.pkl', vectorizer)
    # dump_object_to_pickle_file(domain + '-tfidf-word-min-df-' + str(min_df) + '-max-df-' + str(max_df) + '-word-' + '-'.join(map(str, ngram_range)) + '.pkl', tfidf)


if __name__ == '__main__':
    main()
