import pandas as pd
import jsonlines
import numpy as np
import json
from datasets import load_dataset
import src.utils as utils
from tqdm import tqdm
from ast import literal_eval
import os

if __name__ == "__main__":
    log = utils.get_pylogger(__name__, stdout=True)
else:
    log = utils.get_pylogger(__name__)


def filter_triplets(triplets, entities):
    filtered_triplets = []
    for triplet in triplets:
        try:
            sub = literal_eval(triplet["subject"])["surfaceform"]
        except:
            sub = triplet["subject"]["surfaceform"]
        if sub in entities:
            continue
        try:
            obj = literal_eval(triplet["object"])["surfaceform"]
        except:
            obj = triplet["object"]["surfaceform"]
        if obj in entities:
            continue
        filtered_triplets.append(triplet)
    return filtered_triplets


def filter_entities(sample):
    entities = []
    entity_set = set()
    for triplet in sample["triplets"]:
        try:
            sub = literal_eval(triplet["subject"])
        except:
            sub = triplet["subject"]
        if sub['surfaceform'] in entity_set:
            pass
        else:
            entity_set.add(sub['surfaceform'])
            entities.append(sub)
        try:
            obj = literal_eval(triplet["object"])
        except:
            obj = triplet["object"]
        if obj["surfaceform"] in entity_set:
            pass
        else:
            entity_set.add(obj["surfaceform"])
            entities.append(obj)
    return entities


def filter_relations(sample):
    relations = []
    relation_set = set()
    for triplet in sample["triplets"]:
        try:
            rel = literal_eval(triplet["predicate"])
        except:
            rel = triplet["predicate"]
        if rel["surfaceform"] in relation_set:
            pass
        else:
            relation_set.add(rel["surfaceform"])
            relations.append(rel)
    return relations


def read_data(path, type, split, skip_first_lines, num_samples):
    # data should be in a form of list of dicts
    if path.endswith(".csv"):
        data = pd.read_csv(path).to_dict(orient='records')
    elif path.endswith(".jsonlines"):
        data = []
        with jsonlines.open(path, 'r') as f:
            for line in f:
                data.append(line)
    elif type and split:
        data = load_dataset(path, type, split=split)
    else:
        raise NotImplementedError
    if len(data) - skip_first_lines < num_samples:
        log.warning(f"Number of datapoints read: {len(data)}, skipping first lines: {skip_first_lines}, is smaller than number of samples required: {num_samples}")
    return data


def prepare_synthie_entity_filtered_data(path, save_path, num_samples=100000, skip_first_lines=0, frac_samples_ent_filt=0.3, max_entities_removed=2, type=None, split=None, seed=0):
    data = read_data(path, type, split, skip_first_lines, num_samples)
    np.random.seed(seed)

    data_processed = []
    cnt = 0
    for sample in tqdm(data):
        if cnt < skip_first_lines:
            cnt += 1
            continue
        if np.random.rand() < frac_samples_ent_filt:
            num_to_remove = min(np.random.choice(max_entities_removed) + 1, len(sample["entities"]))
            entities_to_remove = np.random.choice([elem["surfaceform"] for elem in sample["entities"]], num_to_remove, replace=False)
            sample["triplets"] = filter_triplets(sample["triplets"], entities_to_remove)
            sample["entities"] = filter_entities(sample)
            sample["relations"] = filter_relations(sample)
            sample["entities_removed"] = list(entities_to_remove)
        else:
            sample["entities_removed"] = None
        data_processed.append(sample)
        cnt += 1
        if cnt >= skip_first_lines + num_samples:
            break

    with jsonlines.open(save_path, 'w') as f:
        for sample in data_processed:
            f.write(sample)


def extract_synthie_variety_data(path, save_path, num_samples=100000, skip_first_lines=0, frac_samples_ent_filt=0.3, max_entities_removed=2, type=None, split=None, min_triplets=2, max_same=1, fraction_same=None, seed=0):
    data = read_data(path, type, split, skip_first_lines, num_samples)
    np.random.seed(seed)

    if fraction_same:
        print("Fraction set, ignoring min_triplets and max_same arguments")

    data_processed = []
    cnt = 0
    for sample in tqdm(data):
        if cnt < skip_first_lines:
            cnt += 1
            continue
        num_subj, num_triplets = count_num_same_sub_and_triplets(sample)
        if (fraction_same and (num_subj <= fraction_same * num_triplets)) or ((fraction_same is None) and (num_triplets >= min_triplets and num_subj <= max_same)):
            if np.random.rand() < frac_samples_ent_filt:
                num_to_remove = min(np.random.choice(max_entities_removed) + 1, len(sample["entities"]))
                entities_to_remove = np.random.choice([elem["surfaceform"] for elem in sample["entities"]],
                                                      num_to_remove, replace=False)
                sample["triplets"] = filter_triplets(sample["triplets"], entities_to_remove)
                sample["entities"] = filter_entities(sample)
                sample["relations"] = filter_relations(sample)
                sample["entities_removed"] = list(entities_to_remove)
            else:
                sample["entities_removed"] = None
            data_processed.append(sample)
            cnt += 1
            if cnt >= skip_first_lines + num_samples:
                break
        else:
            continue

    with jsonlines.open(save_path, 'w') as f:
        for sample in data_processed:
            f.write(sample)



def read_data_from_dir(dir_path):
    data = []
    id_set = set()
    for filename in os.listdir(dir_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(dir_path, filename)
            with jsonlines.open(file_path, 'r') as f:
                for line in f:
                    if line["id"] in id_set:
                        continue
                    id_set.add(line["id"])
                    data.append(line)
    print(f"Read {len(data)} datapoints")
    return data


def count_num_same_sub_and_triplets(sample):
    cnt_triples = 0
    ent_dict = {}
    for triple in sample["triplets"]:
        try:
            triple['subject'] = literal_eval(triple['subject'])
        except:
            pass
        try:
            triple['predicate'] = literal_eval(triple['predicate'])
        except:
            pass
        try:
            triple['object'] = literal_eval(triple['object'])
        except:
            pass
        cnt_triples += 1
        if triple["subject"]["surfaceform"] in ent_dict:
            ent_dict[triple["subject"]["surfaceform"]] += 1
        else:
            ent_dict[triple["subject"]["surfaceform"]] = 1

    max_ent = -1
    for key, val in ent_dict.items():
        if val > max_ent:
            max_ent = val

    return max_ent, cnt_triples


def process(sample, text=None, random_state=None):
    sample_proc = {}
    if text:
        sample_proc["text"] = text
    else:
        sample_proc["text"] = sample["text"]
    cols = ["id", "triplets", "entities", "relations"]
    for col in cols:
        sample_proc[col] = sample[col]

    random_state.shuffle(sample["triplets"])
    return sample_proc


def extract_curie_data(dir_path, save_path, skip_first_lines=0, min_triplets=2, max_same=1, fraction_same=None, max_index=None, **args):
    if fraction_same:
        print("Fraction set, ignoring min_triplets and max_same arguments")
    random_state = np.random.RandomState(0)
    data = read_data_from_dir(dir_path)
    data_proc = []
    if 'variety' in args:
        print("Extracting varied data")
        data_res = []
        cnt = 0
        for sample in data:
            if not ("synthie_id" in sample):
                continue
            if cnt == args['variety']:
                break
            num_subj, num_triplets = count_num_same_sub_and_triplets(sample)
            if fraction_same:
                if num_subj <= fraction_same * num_triplets:
                    data_proc.append(process(sample, random_state=random_state))
                    cnt += 1
                else:
                    data_res.append(sample)
            else:
                if num_triplets >= min_triplets and num_subj <= max_same:
                    data_proc.append(process(sample, random_state=random_state))
                    cnt += 1
                else:
                    data_res.append(sample)
        if cnt <= len(data):
            data_res.extend(data[cnt:])
        if cnt < args['variety']:
            print(f"Warning: only {cnt} samples with at least {min_triplets} triples and at most {max_same} same subjects found, asked for {args['variety']}")
        data = data_res
    if 'alias' in args:
        print("Extracting alias data")
        cnt = 0
        data_res = []
        for sample in data:
            if cnt == args['alias']:
                break
            if "alias" in sample["instruction_types"]:
                idx = sample["instruction_types"].index("alias")
                if (max_index and idx < max_index) or (max_index is None):
                    text = sample["model_completions"][idx]
                    sample_proc = process(sample, text, random_state=random_state)
                    data_proc.append(sample_proc)
                    cnt += 1
                else:
                    data_res.append(sample)
            else:
                data_res.append(sample)
        if cnt <= len(data):
            data_res.extend(data[cnt:])
        if cnt < args["alias"]:
            print(f"Warning: only {cnt} samples with 'alias' as instruction type found, asked for {args['alias']}")

    with jsonlines.open(save_path, 'w') as f:
        for sample in data_proc:
            f.write(sample)
