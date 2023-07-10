from datasets import load_dataset
import pandas as pd


def convert_ontonotes_dataset(dataset):
    entity_types = {
        0: 'O',
        1: 'CARDINAL',
        2: 'DATE',
        3: 'DATE',
        4: 'PERSON',
        5: 'PERSON',
        6: 'NORP',
        7: 'GPE',
        8: 'GPE',
        9: 'LAW',
        10: 'LAW',
        11: 'ORG',
        12: 'ORG',
        13: 'PERCENT',
        14: 'PERCENT',
        15: 'ORDINAL',
        16: 'MONEY',
        17: 'MONEY',
        18: 'WORK_OF_ART',
        19: 'WORK_OF_ART',
        20: 'FAC',
        21: 'TIME',
        22: 'CARDINAL',
        23: 'LOC',
        24: 'QUANTITY',
        25: 'QUANTITY',
        26: 'NORP',
        27: 'LOC',
        28: 'PRODUCT',
        29: 'TIME',
        30: 'EVENT',
        31: 'EVENT',
        32: 'FAC',
        33: 'LANGUAGE',
        34: 'PRODUCT',
        35: 'ORDINAL',
        36: 'LANGUAGE'
    }

    converted_data = []

    for example in dataset:
        sentence = ""
        for token in example['tokens']:
            if sentence and not token.startswith(("'", ",", "!", ".", "?", "%")):
                sentence += " "
            sentence += token

        entities = []
        current_entity = None

        for i in range(len(example['tokens'])):
            token = example['tokens'][i]
            tag = example['tags'][i]

            if tag == 0:
                current_entity = None
                continue

            entity_type = entity_types[tag]
            if current_entity is None or current_entity['type'] != entity_type:
                current_entity = {
                    'text': token,
                    'type': entity_type,
                    'span_start': sentence.index(token),
                    'span_end': sentence.index(token) + len(token)
                }
                entities.append(current_entity)
            else:
                current_entity['text'] += ' ' + token
                current_entity['span_end'] = sentence.index(token) + len(token)

        if entities:
            converted_example = {
                'sentence': sentence,
                'entities': entities
            }
            converted_data.append(converted_example)

    return converted_data


# Load the ontonotes dataset (in this case, the test split)
ontonotes_dataset = load_dataset("tner/ontonotes5", split="test")

# Convert the dataset using the defined function
converted_ontonotes_dataset = convert_ontonotes_dataset(ontonotes_dataset)
df = pd.DataFrame(converted_ontonotes_dataset)
# df.to_parquet('converted_ontonotes_dataset.parquet')

# Printing part
print(converted_ontonotes_dataset[11]['sentence'])
print(converted_ontonotes_dataset[11]['sentence'][48:54])
print(converted_ontonotes_dataset[11]['entities'])
