import pandas as pd
import numpy as np
import warnings
import torch

from mido import MidiFile
from os import path, listdir
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# filter warnings

warnings.filterwarnings('ignore')

class LofiMidiDataset(Dataset):
  def __init__(self, input_seq, target_seq):
    self.input_seqs = input_seq
    self.output_seqs = target_seq
  
  def __len__(self):
    return len(self.input_seqs)
  
  def __getitem__(self, index):
    xs = torch.from_numpy(self.input_seqs[index]).float()
    ys = torch.from_numpy(self.output_seqs[index]).float()
    
    return xs, ys

def read_midi_dataframe(dir_path):
  print("[!] Parsing MIDI into Dataframe")
  
  dir_contents = listdir(dir_path)
  data_arr = []
  data_arr = []

  for file in tqdm(dir_contents):
    # read midi file
    midi_file_path = path.join(dir_path, file)
    midi_data = MidiFile(midi_file_path, clip=True)
  
    for track in midi_data.tracks:
      for message in track:
        if message.is_meta:
          continue
        
        msg_dict = message.dict()
        data_arr.append({
          "status": msg_dict.get('type'),
          "time": msg_dict.get('time'),
          "note": msg_dict.get('note'),
          "velocity": msg_dict.get('velocity'),
          "channel": msg_dict.get('channel'),
          "file": file
        })
  return pd.DataFrame(data_arr)

# cleaning functions

def clean_dataframe(df: pd.DataFrame):
  
  df = df.fillna(0)

  df = df[df['status'] != 'control_change']

  # convert note state to integer

  df['status_int'] = df['status'].apply(lambda x: int(x == 'note_on'))

  # remove all the samples with channel != 0
  df = df[df['channel'] == 0]

  # map all the time values between 0 to 100
  df['time'] = df['time'].apply(lambda x: 100 if x > 100 else x)
  
  # map all the velocities between 0 to 127
  df['velocity'] = df['velocity'].apply(lambda x: 127 if x > 127 else x)
  
  # remove all nan vals

  return df  

# create time sequences

def create_time_sequences(df: pd.DataFrame, seq_len = 4):


  print("[!] Generating Time Sequences")
    
  input_sequences = []
  target_sequence = []
  
  # read all the song names from the df
  
  file_names_unique = list(df['file'].unique())
  
  for file in tqdm(file_names_unique):
    # select all rows with same file name
    file_rows = df[df['file'] == file][['status_int', 'note', 'time', 'velocity']]
    
    # create sequences
    
    for row in range(0, len(file_rows.values)-seq_len):
      input_seq = file_rows.iloc[row:row+seq_len].values
      target_seq = file_rows.iloc[row + seq_len].values

      input_sequences.append(input_seq)
      target_sequence.append(target_seq)
      

  input_sequences = np.array(input_sequences)
  target_sequence = np.array(target_sequence)
  
  return input_sequences, target_sequence

def pipeline(midi_dir_path: str):

  df = read_midi_dataframe(midi_dir_path)
  df = clean_dataframe(df)
  
  input_seq, target_seq = create_time_sequences(df)
  
  train_dataset = LofiMidiDataset(input_seq, target_seq)
  train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=False)
  
  return train_dataloader
