import spacy as sp
import random
from spacy.tokens import DocBin


nlp = sp.load("en_core_web_sm") 

#text = "find temperature and humidity hourly and daily using the date 2024-12-20"


#Train the model to know the entity is map to which label
training_data = [
    ("find temperature for weekly", {"entities": [(5,16,"attr"),(21,27,"time")]}),
    ("find door for hourly on 2024-12-20", {"entities": [(5,9,"attr"),(14,20,"time"),(24,34,"date")]}),

    ("find humidity for monthly", {"entities": [(5,13,"attr"),(18,25,"time")]}),
    ("find temperature for daily", {"entities": [(5,16,"attr"),(21,26,"time")]}),

    ("find motion for weekly", {"entities": [(5,11,"attr"),(16,22,"time")]}),
    ("find humidity for hourly on 2024-12-20", {"entities": [(5,13,"attr"),(18,24,"time"),(28,38,"date")]}),

    ("find door for monthly", {"entities": [(5,9,"attr"),(14,21,"time")]}),
    ("find motion for daily", {"entities": [(5,11,"attr"),(16,21,"time")]}),
    
    ("find door and motion for weekly and monthly", {"entities": [(5,9,"attr"),(14,20,"attr"),(25,31,"time"),(36,43,"time")]}),
    ("find humidity and temperature for daily and weekly", {"entities": [(5,13,"attr"),(18,29,"attr"),(34,39,"time"),(44,50,"time")]}),
    ("find motion, door, and humidity for hourly and daily on 2025-06-15", {"entities": [(5,11,"attr"),(13,17,"attr"),(23,31,"attr"),(36,42,"time"),(47,52,"time"),(56,66,"date")]}),
    ("find temperature and door for weekly and monthly on 2025-07-20", {"entities": [(5,16,"attr"),(21,25,"attr"),(30,36,"time"),(41,48,"time"),(52,62,"date")]}),
    ("find humidity, motion, and door for daily and yearly", {"entities": [(5,13,"attr"),(15,21,"attr"),(27,31,"attr"),(36,41,"time"),(46,52,"time")]}),
    ("find motion and temperature for hourly and weekly on 2025-08-10", {"entities": [(5,11,"attr"),(16,27,"attr"),(32,38,"time"),(43,49,"time"),(53,63,"date")]}),
    ("find temperature, door, and humidity for daily and monthly", {"entities": [(5,16,"attr"),(18,22,"attr"),(28,36,"attr"),(41,46,"time"),(51,58,"time")]})
]


"""
    ("find door and expected frequency for hourly on 2024-12-20", {"entities": [(9,27,"ef")]}),
    ("find motion and expected frequency for monthly", {"entities": [(11,29,"ef")]}),
    ("find door and expected frequency for daily", {"entities": [(9,27,"ef")]}),

    ("find temperature and humidity hourly and daily using the date 2024-12-20", {"entities": [(5,16,"attr")]}),
    ("find temperature and humidity hourly and daily using the date 2024-12-20", {"entities": [(21,29,"attr")]}),
    ("find temperature and humidity hourly and daily using the date 2024-12-20", {"entities": [(30,36,"time")]}),
    ("find temperature and humidity hourly and daily using the date 2024-12-20", {"entities": [(41,46,"time")]}),
    ("find temperature and humidity hourly and daily using the date 2024-12-20", {"entities": [(62,72,"date")]}),

    ("find motion and its expected frequency monthly and weekly using the date 2024-12-16", {"entities": [(5,11,"attr")]}),
    ("find door and its expected frequency monthly and weekly using the date 2024-12-16", {"entities": [(5,9,"attr")]}),
    ("find motion and its expected frequency monthly and weekly using the date 2024-12-16", {"entities": [(20,38,"ef")]}),
    ("find door and its expected frequency monthly and weekly using the date 2024-12-16", {"entities": [(18,36,"ef")]}),

    ("find motion and its expected frequency monthly and weekly using the date 2024-12-16", {"entities": [(39,46,"time")]}),
    ("find door and its expected frequency monthly and weekly using the date 2024-12-16", {"entities": [(37,44,"time")]}),
    ("find motion and its expected frequency monthly and weekly using the date 2024-12-16", {"entities": [(51,57,"time")]}),
    ("find door and its expected frequency monthly and weekly using the date 2024-12-16", {"entities": [(49,55,"time")]}),
    ("find motion and its expected frequency monthly and weekly using the date 2024-12-16", {"entities": [(73,83,"date")]}),
    ("find door and its expected frequency monthly and weekly using the date 2024-12-16", {"entities": [(71,81,"date")]}),"""


db = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            ents.append(span)
        else:
            print(f"Warning: Invalid span for '{label}' in text '{text}' from {start} to {end}")
    
    if ents:
        doc.ents = ents
    else:
        print(f"Warning: No valid entities for text '{text}'")
    db.add(doc)
db.to_disk("./train.spacy")

