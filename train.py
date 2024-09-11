from data_loader import pipeline
from model import LofiRNN


MIDI_DIR_PATH = "./data"

train_dataloader = pipeline(midi_dir_path=MIDI_DIR_PATH)


model = LofiRNN()
print(model)

model.train_model(train_df=train_dataloader)