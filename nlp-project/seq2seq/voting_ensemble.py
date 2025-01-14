# from gensim.models import KeyedVectors
import Levenshtein
import glob
import json

import numpy as np

from fire import Fire

# def get_sentence_vector(sent, model):
#     raise NotImplementedError
#     sent_vec = []
#     for word in sent:
#         vec = model[word]
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
#     return sent_vec


def voting_ensemble(gold_data : str,pred_data_path : str, out_dir : str):
    #
    # Prepare data
    #
    gold_data_dics = []
    for line in open(gold_data):
        data_dic = json.loads(line)
        gold_data_dics.append(data_dic)

    pred_data_dics_list = []
    paths = glob.glob(pred_data_path+'/*.jsonl')
    for path in paths:
        pred_data_dics = []
        for line in open(path):
            data_dic = json.loads(line)
            pred_data_dics.append(data_dic)
        pred_data_dics_list.append(pred_data_dics)

    #
    # Ensemble
    #
    all_spoilers = []
    for pred_data_dics in pred_data_dics_list:
        spoilers = []
        for pred_data in pred_data_dics:
            pred_spoiler = pred_data['spoiler']
            spoilers.append(pred_spoiler)
        all_spoilers.append(spoilers)

    ensemble_data_dics = []
    for i in range(len(spoilers)):
        spoilers_i = [spoilers[i] for spoilers in all_spoilers]
        all_l_dist = []
        for j in range(len(spoilers_i)):
            l_dist_list = []
            target = spoilers_i[j]
            for k in range(len(spoilers_i)):
                if j == k:
                    continue
                else:
                    other_spoiler = spoilers_i[k]
                l_dist = Levenshtein.distance(target, other_spoiler)
                l_dist_list.append(l_dist)
            all_l_dist.append(l_dist_list)

        sum_l_dist = [sum(l_dist_list) for l_dist_list in all_l_dist]
        selected_idx = np.argmin(sum_l_dist)
        selected_spoiler = spoilers_i[selected_idx]
        # print(spoilers_i)
        # print(all_l_dist)
        # print(selected_idx)
        # print(selected_spoiler)
        # print()

        ensemble_data = {'uuid': gold_data_dics[i]['uuid'],
                         'spoiler': selected_spoiler
                         }
        ensemble_data_dics.append(ensemble_data)

    with open(out_dir, 'w') as fo:
        for d in ensemble_data_dics:
            fo.write(f'{json.dumps(d)}\n')


if __name__ == '__main__':
    Fire(voting_ensemble)