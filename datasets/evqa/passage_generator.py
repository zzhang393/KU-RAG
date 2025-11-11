import os, sys
import json
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from image_add_caption import add_caption
from images_merger import merge



def find_duplicate_indices(input_list):
    index_map = {}
    duplicates = []

    for index, value in enumerate(input_list):
        if value in index_map:
            # If element already exists in dictionary, it's a duplicate
            if value not in duplicates:
                duplicates.append(value)  # Record duplicate only once
        else:
            index_map[value] = [index]

    # Find all indices of duplicate elements
    result = []
    for value in duplicates:
        indices = [i for i, x in enumerate(input_list) if x == value]
        result.extend(indices)
    return result

if __name__ == '__main__':
    # Load questions
    print('Questions loading...')
    file_path = './E-VQA_data.csv'
    idx_result_file = './results/idx_result.json'
    text_metadata_file = './event_text_index.csv'

    text_metadata = []
    img_metadata = []
    questions_list = []
    idx_result = []

    # img_set = set()
    df = pd.read_csv(file_path, header=0)
    questions_list = df.to_dict('records')

    with open(idx_result_file, 'r') as file:
        idx_results = json.load(file)

    text_metadata = pd.read_csv(text_metadata_file, header=0).values.tolist()

    for i, (question, idx_result) in enumerate(zip(questions_list, idx_results)):
        img0 = f"./image/{question['image_id']}.jpg"
        save_path = f"./image_test/{question['question_id']}.jpg"

        img_list = []
        img_list0 = []
        dup_list = []
        # img0 = add_caption(img0, '')
        # img_list.append(img0)
        print(question['question_id'])

        if len(idx_results) != 0:
            for sub_idx in idx_result:
                img_list0.append(sub_idx['image'])

            dup_list = find_duplicate_indices(img_list0)

            if len(dup_list) == 0:
                for j, sub_idx in enumerate(idx_result):
                    cpt = ''
                    cpt1 = ''

                    idx = int(sub_idx['idx'])
                    cpt0 = '[' + sub_idx['name'] + ']: '
                    cpt2 = text_metadata[idx][2]
                    if isinstance(cpt, list):
                        cpt2 = cpt2[0].strip("\"'")
                        cpt1 = cpt1 + cpt2
                    else:
                        cpt2 = cpt2.strip("[]").strip("\"'")
                        cpt1 = cpt1 + cpt2
                    cpt = cpt0 + cpt1
                    img = f"./image/{sub_idx['image']}.jpg"
                    img_cpt = add_caption(img, cpt)
                    img_list.append(img_cpt)
            else:
                cpt = ''
                cpt1 = ''
                img_temp = ''
                for j_idx in dup_list:
                    sub_idx = idx_result[j_idx]
                    img_temp = sub_idx['image']
                    idx = int(sub_idx['idx'])
                    cpt0 = '[' + sub_idx['name'] + ']: '
                    cpt2 = text_metadata[idx][2]
                    if isinstance(cpt, list):
                        cpt2 = cpt2[0].strip("\"'")
                        cpt1 = cpt1 + cpt2
                    else:
                        cpt2 = cpt2.strip("[]").strip("\"'")
                        cpt1 = cpt1 + cpt2
                cpt = cpt0 + cpt1
                img = f"./image/{img_temp}.jpg"
                img_cpt = add_caption(img, cpt)
                img_list.append(img_cpt)

                for j, sub_idx in enumerate(idx_result):
                    cpt = ''
                    cpt1 = ''
                    if j not in dup_list:
                        idx = int(sub_idx['idx'])
                        cpt0 = '[' + sub_idx['name'] + ']: '
                        cpt2 = text_metadata[idx][2]
                        if isinstance(cpt, list):
                            cpt2 = cpt2[0].strip("\"'")
                            cpt1 = cpt1 + cpt2
                        else:
                            cpt2 = cpt2.strip("[]").strip("\"'")
                            cpt1 = cpt1 + cpt2
                        cpt = cpt0 + cpt1
                        img = f"./image/{sub_idx['image']}.jpg"
                        img_cpt = add_caption(img, cpt)
                        img_list.append(img_cpt)

        if len(img_list) != 0:
            merge(img_list, save_path)





