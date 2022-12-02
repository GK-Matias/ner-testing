import spacy
from typing import Dict
import random
import string


model = spacy.load("./model1/")  # type: ignore
model2 = spacy.load("./model2/")  # type: ignore

# Inferencing
for i in range(500):
    random_strings = []
    for j in range(1,i%5+2):
        random_string=''.join(random.Random(i*10+j).choices(string.ascii_lowercase, k=5))
        random_strings.append(random_string)
    test_data=' '.join(random_strings)

    doc: spacy.tokens.doc.Doc = model(test_data)
    result_dictionary: Dict[str, str] = {}
    for ent in doc.ents:
        result_dictionary[str(ent.label_)] = str(ent)

    doc2: spacy.tokens.doc.Doc = model2(test_data)
    result_dictionary2: Dict[str, str] = {}
    for ent in doc2.ents:
        result_dictionary2[str(ent.label_)] = str(ent)
    if result_dictionary != result_dictionary2:
        print("error")
        print(test_data)
        print(result_dictionary)
        print(result_dictionary2)
