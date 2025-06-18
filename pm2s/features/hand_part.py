import torch

from pm2s.features._processor import MIDIProcessor
from pm2s.models.hand_part import RNNHandPartModel
from pm2s.constants import model_state_dict_paths

class RNNHandPartProcessor(MIDIProcessor):

    def __init__(self, state_dict_path=None, device=torch.device('cuda')):
        if state_dict_path is None:
            state_dict_path = model_state_dict_paths['hand_part']['state_dict_path']
        self.device = device
        zenodo_path = model_state_dict_paths['hand_part']['zenodo_path']

        self._model = RNNHandPartModel()
        self.load(state_dict_path=state_dict_path, zenodo_path=zenodo_path)
        self._model.to(self.device)

    def process_note_seq(self, note_seq):
        # Process note sequence

        x = torch.tensor(note_seq).unsqueeze(0).to(self.device)

        # Forward pass
        hand_probs = self._model(x)

        # Post-processing
        hand_probs = hand_probs.squeeze(0).detach().cpu().numpy()
        hand_parts = (hand_probs > 0.5).astype(int)

        return hand_parts
    