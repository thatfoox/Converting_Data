from datasets import load_dataset
import pandas as pd

def convert_conll_dataset(dataset):
    entity_type_mapping = {
        0: 'O',
        1: 'PER',
        2: 'PER',
        3: 'ORG',
        4: 'ORG',
        5: 'LOC',
        6: 'LOC',
        7: 'MISC',
        8: 'MISC'
    }

    converted_dataset = []

    for example in dataset:
        sentence = ""
        for t in example['tokens']:
            if sentence and not t.startswith(("'", ",", "!", ".", "?", "%")):
                sentence += " "
            sentence += t
        entities = convert_entities(example['tokens'], example['ner_tags'], sentence, entity_type_mapping)
        converted_example = {
            'sentence': sentence,
            'entities': entities
        }
        converted_dataset.append(converted_example)

    return converted_dataset


def convert_entities(tokens, ner_tags, sentence, entity_type_mapping):
    entities = []
    prev_entity = None
    token_start = 0

    for i in range(len(tokens)):
        token = tokens[i]
        ner_tag = ner_tags[i]
        if ner_tag != 'O' and ner_tag != 0:
            entity_type = entity_type_mapping[ner_tag]

            if prev_entity is not None and entity_type == prev_entity['type']:
                prev_entity['text'] += " " + token
                prev_entity['span_end'] = sentence.index(token, prev_entity['span_start']) + len(token)
                continue

            entity = {
                'text': token,
                'type': entity_type,
                'span_start': sentence.index(token, token_start),
                'span_end': sentence.index(token, token_start) + len(token)
            }
            entities.append(entity)
            prev_entity = entity
            token_start = entity['span_end'] + 1
        else:
            prev_entity = None
    return entities


# Loading the conll2003  dataset (in this case test)
conll_dataset = load_dataset("conll2003", split="test")


# Converting the dataset using the defined functions
converted_conll_dataset = convert_conll_dataset(conll_dataset)
df = pd.DataFrame(converted_conll_dataset)
#df.to_parquet("converted_conll_dataset_test.parquet")

# Printing part
# print(converted_conll_dataset[-2])
print(converted_conll_dataset[8]['sentence'])
print(converted_conll_dataset[8]['sentence'][53:63])
print(converted_conll_dataset[8]['entities'])
