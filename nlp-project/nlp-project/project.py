import argparse
import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--king',
                        type=str,
                        help='Path to John King\'s JSONL output file',
                        required=True)
    parser.add_argument('--spellman',
                        type=str,
                        help='Path to Sabrina Spellman\'s JSONL output file',
                        required=True)
    parser.add_argument('--fosdick',
                        type=str,
                        help='Path to Mr. Fodick\'s JSONL output file',
                        required=True)
    parser.add_argument('--input',
                        type=str,
                        help='Path to the JSONL input file',
                        required=True)
    parser.add_argument('--output',
                        type=str,
                        help='Path to where to put the outputted JSONL file',
                        required=True)
    return parser.parse_args()


def cos_sim(john_king_path, sabrina_spellman_path, mr_fosdick_path, input_path):
    john_king = pd.read_json(path_or_buf=john_king_path, lines=True)
    sabrina_spellman = pd.read_json(path_or_buf=sabrina_spellman_path, lines=True)
    mr_fosdick = pd.read_json(path_or_buf=mr_fosdick_path, lines=True)
    test = pd.read_json(path_or_buf=input_path, lines=True)

    all_text = []
    for texts in test['targetParagraphs']:
        for text in texts:
            all_text.append(text)

    # Prepare a TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit_transform(all_text)

    results = pd.DataFrame(columns=['uuid', 'spoiler'])

    # Counts are in the order [phrase, passage, multi]
    john_king_count = [0, 0, 0]
    sabrina_spellman_count = [0, 0, 0]
    mr_fosdick_count = [0, 0, 0]

    for index, row in test.iterrows():
        uuid = row['uuid']
        jk = john_king.uuid[john_king.uuid == uuid].index.tolist()[0]
        ss = sabrina_spellman.uuid[sabrina_spellman.uuid == uuid].index.tolist()[0]
        mf = mr_fosdick.uuid[mr_fosdick.uuid == uuid].index.tolist()[0]

        jk_spoiler = john_king.iloc[jk]['spoiler']
        ss_spoiler = sabrina_spellman.iloc[ss]['spoiler']
        mf_spoiler = mr_fosdick.iloc[mf]['spoiler']

        if type(jk_spoiler) is str:
            jk_spoiler = [jk_spoiler]
        if type(ss_spoiler) is str:
            ss_spoiler = [ss_spoiler]
        if type(mf_spoiler) is str:
            mf_spoiler = [mf_spoiler]

        # Vectorize
        jk_vec = vectorizer.transform(jk_spoiler)
        ss_vec = vectorizer.transform(ss_spoiler)
        mf_vec = vectorizer.transform(mf_spoiler)
        input_vec = vectorizer.transform(row['targetParagraphs'])

        # Calculate cosine similarities
        sim_jk = cosine_similarity(input_vec, jk_vec).max()
        sim_ss = cosine_similarity(input_vec, ss_vec).max()
        sim_mf = cosine_similarity(input_vec, mf_vec).max()

        tag = row['tags'][0]
        if sim_jk >= sim_ss and sim_jk >= sim_mf:
            results.loc[len(results)] = john_king.iloc[jk]
            if tag == 'phrase':
                john_king_count[0] += 1
            elif tag == 'passage':
                john_king_count[1] += 1
            else:
                john_king_count[2] += 1
        elif sim_ss >= sim_jk and sim_ss > sim_mf:
            results.loc[len(results)] = sabrina_spellman.iloc[ss]
            if tag == 'phrase':
                sabrina_spellman_count[0] += 1
            elif tag == 'passage':
                sabrina_spellman_count[1] += 1
            else:
                sabrina_spellman_count[2] += 1
        else:
            results.loc[len(results)] = mr_fosdick.iloc[jk]
            if tag == 'phrase':
                mr_fosdick_count[0] += 1
            elif tag == 'passage':
                mr_fosdick_count[1] += 1
            else:
                mr_fosdick_count[2] += 1

    return results, john_king_count, sabrina_spellman_count, mr_fosdick_count


def print_jsonl(results, output_path, john_king, sabrina_spellman, mr_fosdick):
    results.to_json(output_path, orient='records', lines=True)

    print('John King phrase count: ' + str(john_king[0]))
    print('John King phrase count: ' + str(john_king[1]))
    print('John King phrase count: ' + str(john_king[2]))
    print('Sabrina Spellman phrase count: ' + str(sabrina_spellman[0]))
    print('Sabrina Spellman phrase count: ' + str(sabrina_spellman[1]))
    print('Sabrina Spellman phrase count: ' + str(sabrina_spellman[2]))
    print('Mr. Fosdick phrase count: ' + str(mr_fosdick[0]))
    print('Mr. Fosdick phrase count: ' + str(mr_fosdick[1]))
    print('Mr. Fosdick phrase count: ' + str(mr_fosdick[2]))


if __name__ == '__main__':
    args = parse_args()
    output, john_king_count, sabrina_spellman_count, mr_fosdick_count = cos_sim(args.king,
                                                                                          args.spellman,
                                                                                          args.fosdick,
                                                                                          args.input)
    print_jsonl(output, args.output, john_king_count, sabrina_spellman_count, mr_fosdick_count)
