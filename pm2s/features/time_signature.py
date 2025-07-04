import torch
import numpy as np

from pm2s.features._processor import MIDIProcessor
from pm2s.models.time_signature import CNNTimeSignatureModel
from pm2s.io.midi_read import read_note_sequence
from pm2s.constants import model_state_dict_paths

class CNNTimeSignatureProcessor(MIDIProcessor):

    def __init__(self, state_dict_path=None, device=torch.device('cuda')):
        if state_dict_path is None:
            state_dict_path = model_state_dict_paths['time_signature']['state_dict_path']
        self.device = device
        zenodo_path = model_state_dict_paths['time_signature']['zenodo_path']

        self._model = CNNTimeSignatureModel()
        self.load(state_dict_path=state_dict_path, zenodo_path=zenodo_path)
        self._model.to(self.device)

    def process_note_seq(self, note_seq):
        # Process note sequence

        x = torch.tensor(note_seq).unsqueeze(0).to(self.device)

        # Forward pass
        ts_probs = self._model(x)

        # Post-processing
        ts_probs = ts_probs.squeeze(0).detach().cpu().numpy()  # (seq_len,)
        onsets = note_seq[:, 1]
        time_signature = self.pps(ts_probs, onsets)

        return time_signature
    
    def pps(self, ts_probs, onsets):
        # ts_probs: (seq_len,)
        # onsets: (seq_len,)
        ts = (ts_probs > 0.5).astype(int)
        ts = np.bincount(ts).argmax()
        ts = '4/4' if ts == 0 else '3/4'
        return ts
