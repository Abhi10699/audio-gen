import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

class LofiRNN(nn.Module):
  def __init__(self, in_feats = 4,lstm_hidden_size = 64):
    super().__init__()
    
    self.fc1 = nn.LSTM(in_feats, lstm_hidden_size, batch_first=True)
    
    # outputs
    self.note_on_off = nn.Linear(lstm_hidden_size, 1)
    self.velocity = nn.Linear(lstm_hidden_size, 1)
    self.delay = nn.Linear(lstm_hidden_size, 1)
    self.pitch = nn.Linear(lstm_hidden_size, 127)

    self.to(device=DEVICE)

  def forward(self, x):
    """
    x = [[on/off, delay, key, velocity]]
    """
    _, (h_n, _) = self.fc1(x)
    feature = h_n.squeeze()
    
    on_off = F.sigmoid(self.note_on_off(feature))
    pitch = self.pitch(feature)
    velocity = self.velocity(feature)
    delay = self.delay(feature)
    
    return on_off, pitch, velocity, delay

  def train_model(self, train_df, epochs = 100):

    self.train()

    optimizer = optim.Adam(self.parameters())
    overall_loss_arr = []
    
    for _ in range(epochs):
              
      avg_overall_loss = 0.0
      avg_pitch_loss = 0.0
      avg_vel_loss = 0.0
      avg_delay_loss = 0.0
      avg_on_off_loss = 0.0
    
      
      for _, sample in enumerate(train_df):
        optimizer.zero_grad()
        
        xs = sample[0].to(DEVICE)
        ys = sample[1].to(DEVICE)
                        
        output = self.forward(xs)
        on_off, pitch, vel, delay = output
        
        on_off_loss = F.binary_cross_entropy(on_off, ys[:, 0].unsqueeze(1))
        pitch_loss = F.cross_entropy(pitch, ys[:, 1].long())
        
        vel_loss = F.mse_loss(vel, ys[:, 2].unsqueeze(1)) * 0.05
        delay_loss = F.mse_loss(delay, ys[:, 2].unsqueeze(1)) * 0.05
        
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        
        overall_loss = on_off_loss + pitch_loss + vel_loss + delay_loss
        overall_loss.backward()
        optimizer.step()
        
        avg_overall_loss += overall_loss.item()
        avg_pitch_loss += pitch_loss.item()
        avg_vel_loss += vel_loss.item()
        avg_delay_loss += delay_loss.item()
        avg_on_off_loss += on_off_loss.item()

      print(f"On_Off_loss: {avg_on_off_loss}, pitch_loss: {avg_pitch_loss}, velocity_loss: {avg_vel_loss}, delay_loss: {avg_delay_loss}")
      overall_loss_arr.append(avg_overall_loss)
      