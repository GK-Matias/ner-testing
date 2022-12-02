import spacy
from typing import List, Dict,Tuple
from spacy.util import minibatch
from spacy.training import Example
import numpy
import random
import string

### Remove randomness from spacy ###
numpy.random.seed(0)
spacy.util.fix_random_seed(0)
### ---------------------------- ###

train_data = [
    ('1 2 ETG TILLUFT VENT', {'entities': [(0, 1, 'D'), (2, 3, 'D'), (4, 7, 'D'), (8, 15, 'F'), (16, 20, 'C')]}),
    ('1 ETG BYGG 3 1 11 V K WC', {'entities': [(0, 1, 'D'), (2, 5, 'D'), (6, 10, 'A'), (11, 12, 'A'), (13, 14, 'D'), (15, 17, 'E'), (18, 19, 'E'), (20, 21, 'E'), (22, 24, 'E')]}),
    ('1 ETG BYGG 3 1 13 GARDEROBE', {'entities': [(0, 1, 'D'), (2, 5, 'D'), (6, 10, 'A'), (11, 12, 'A'), (13, 14, 'D'), (15, 17, 'E'), (18, 27, 'E')]}),
    ('1 ETG BYGG 3 1 14 KORRobjectIdOR', {'entities': [(0, 1, 'D'), (2, 5, 'D'), (6, 10, 'A'), (11, 12, 'A'), (13, 14, 'D'), (15, 17, 'E'), (18, 32, 'E')]}),
    ('1 ETG BYGG 3 1 18 FELLESROM', {'entities': [(0, 1, 'D'), (2, 5, 'D'), (6, 10, 'A'), (11, 12, 'A'), (13, 14, 'D'), (15, 17, 'E'), (18, 27, 'E')]}),
    ('1 ETG GYMSAL', {'entities': [(0, 1, 'D'), (2, 5, 'D'), (6, 12, 'E')]}),
    ('1 ETG HALL', {'entities': [(0, 1, 'D'), (2, 5, 'D'), (6, 10, 'E')]}),
    ('1 ETG SCENE', {'entities': [(0, 1, 'D'), (2, 5, 'D'), (6, 11, 'E')]}),
    ('1 etg 107 WC jenter', {'entities': [(0, 1, 'D'), (2, 5, 'D'), (6, 9, 'E'), (10, 12, 'E'), (13, 19, 'E')]}),
    ('1002 POS MV', {'entities': [(0, 4, 'E'), (5, 8, 'G'), (9, 11, 'G')]}),
    ('1002 RYc PV', {'entities': [(0, 4, 'E'), (5, 8, 'G'), (9, 11, 'G')]}),
    ('1002 RYc SP', {'entities': [(0, 4, 'E'), (5, 8, 'G'), (9, 11, 'G')]}),
    ('1002 RYt SPD', {'entities': [(0, 4, 'E'), (5, 8, 'G'), (9, 12, 'G')]}),
    ('1002 RYt SPR', {'entities': [(0, 4, 'E'), (5, 8, 'G'), (9, 12, 'G')]}),
    ('1002 SR401 POS MV', {'entities': [(0, 4, 'E'), (5, 10, 'F'), (11, 14, 'G'), (15, 17, 'G')]}),
    ('1002 SR401 VOL MV', {'entities': [(0, 4, 'E'), (5, 10, 'F'), (11, 14, 'G'), (15, 17, 'G')]}),
    ('1002 SR501 POS C', {'entities': [(0, 4, 'E'), (5, 10, 'F'), (11, 14, 'G'), (15, 16, 'G')]}),
    ('1002 SR501 POS MV', {'entities': [(0, 4, 'E'), (5, 10, 'F'), (11, 14, 'G'), (15, 17, 'G')]}),
    ('1021 SR401 POS C', {'entities': [(0, 4, 'E'), (5, 10, 'F'), (11, 14, 'G'), (15, 16, 'G')]}),
    ('1021 SR401 POS MV', {'entities': [(0, 4, 'E'), (5, 10, 'F'), (11, 14, 'G'), (15, 17, 'G')]})
]
max_itn=1000
batchsize = 8
dropout=0.35
labels_array = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
]
model_directory = "./model1/"

# Initializing model
nlp: spacy.lang.en.English = spacy.blank("en")
nlp.add_pipe("ner", last=True)
ner = nlp.get_pipe("ner")
for label in labels_array:
    ner.add_label(label)  # type: ignore
optimizer = nlp.initialize()
pipe_exceptions: List[str] = ["ner"]
other_pipes: List[str] = [
    pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions
]

# training
with nlp.select_pipes(disable=[*other_pipes]):
    for itn in range(max_itn):
        losses: Dict[str, float] = {}
        # batch up the examples using spaCy's minibatch
        batch: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]
        for batch in minibatch(train_data, size=batchsize):
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=dropout, sgd=optimizer, losses=losses)

# save model to disk if you want to benchmark two models
# nlp.to_disk(model_directory)


# Inferencing
for i in range(500):
    random_strings = []
    for j in range(1,i%5+2):
        random_string=''.join(random.Random(i*10+j).choices(string.ascii_lowercase, k=5))
        random_strings.append(random_string)
    test_data=' '.join(random_strings)
    doc: spacy.tokens.doc.Doc = nlp(test_data)
    result_dictionary: Dict[str, str] = {}
    for ent in doc.ents:
        result_dictionary[str(ent.label_)] = str(ent)
    print(test_data)
    print(result_dictionary)