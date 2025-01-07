import spacy as sp
import random
from spacy.tokens import DocBin


nlp = sp.load("en_core_web_sm") 

#text = "find temperature and humidity hourly and daily using the date 2024-12-20"


#Train the model to know the entity is map to which label
training_data = [
    ("find humidity for weekly", {"entities": [(5,13,"attr"),(18,24,"time")]}),
    ("find temperature for hourly on 2024-12-20", {"entities": [(5,16,"attr"),(21,27,"time"),(31,41,"date")]}),
    ("find temperature for monthly", {"entities": [(5,16,"attr"),(21,28,"time")]}),
    ("find door for daily", {"entities": [(5,9,"attr"),(14,19,"time")]}),
    ("find motion for monthly", {"entities": [(5,11,"attr"),(16,23,"time")]}),
    ("find humidity for daily", {"entities": [(5,13,"attr"),(18,23,"time")]}),
    ("find door for weekly", {"entities": [(5,9,"attr"),(14,20,"time")]}),
    ("find motion for hourly on 2024-12-20", {"entities": [(5,11,"attr"),(16,22,"time"),(26,36,"date")]}),


    ("find temperature and humidity for daily and weekly on 2025-01-10", {"entities": [(5,16,"attr"),(21,29,"attr"),(34,39,"time"),(44,50,"time"),(54,64,"date")]}),
    ("find motion and door for hourly and daily on 2025-02-15", {"entities": [(5,11,"attr"),(16,20,"attr"),(25,31,"time"),(36,41,"time"),(45,55,"date")]}),
    ("find door and humidity for weekly and monthly on 2025-03-20", {"entities": [(5,9,"attr"),(14,22,"attr"),(27,33,"time"),(38,45,"time"),(49,59,"date")]}),
    ("find temperature and motion for daily and monthly on 2025-04-25", {"entities": [(5,16,"attr"),(21,27,"attr"),(32,37,"time"),(42,49,"time"),(53,63,"date")]}),
    ("find humidity and door for hourly and weekly", {"entities": [(5,13,"attr"),(18,22,"attr"),(27,33,"time"),(38,44,"time")]}),
    ("find motion and temperature for daily and yearly on 2025-05-30", {"entities": [(5,11,"attr"),(16,27,"attr"),(32,37,"time"),(42,48,"time"),(52,62,"date")]}),
    ("find temperature, motion and humidity for hourly and daily", {"entities": [(5,16,"attr"),(18,24,"attr"),(29,37,"attr"),(42,48,"time"),(53,58,"time")]})
]

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
db.to_disk("./dev.spacy")


