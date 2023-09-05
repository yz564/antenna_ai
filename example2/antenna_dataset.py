from typing import Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from antenna_parameters import AntennaParameters
from convert_to_im import convert_batch_to_im, plot_antenna

import torchvision.transforms as transforms
torch.set_printoptions(profile="full")

def get_num_resonances(pred: np.ndarray):
	minima = 0
	for i in range(1, len(pred) - 1):
		if pred[i] <= pred[i-1] and pred[i] <= pred[i+1] and 20 * np.log10(pred[i]) < -3:
			minima += 1
	return minima  

class SimulationDataSet(Dataset):
    def __init__(self, data_file: str, antenna_parameters: AntennaParameters):
        raw = np.loadtxt(data_file, delimiter=',', dtype=float)
        self._freqs = raw[0, 12:713:10]
        self._params = raw[1:, :12]
        self._preds = raw[1:, 12:713:10]
        self._targets = antenna_parameters.targets
        self._values = antenna_parameters.calculate_score(self._preds)

        resonances = []

        for pred in self._preds:
            minima = get_num_resonances(pred)
            resonances.append(minima)

        self.resonances = np.array(resonances)
        self.pred_weights = np.zeros_like(self._preds)

        # 1D convolution to get resonance mask
        pad = 3
        padding = np.zeros((self._preds.shape[0], pad))
        padded_preds = np.concatenate((padding, -1 *  np.log10(self._preds), padding), axis=1)
        for i in range(pad, 71 + pad): 
            conv = padded_preds[:, i - pad : i + pad + 1]
            self.pred_weights[:, i - pad] = np.sum(conv, axis=1)


    @property
    def freqs(self) -> np.ndarray:
        return self._freqs

    @property
    def xs(self) -> np.ndarray:
        return self._params

    @property
    def preds(self) -> np.ndarray:
        return self._preds

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def targets(self) -> List[Tuple[float]]:
        return self._targets

    def add_images(self, im_file: str, augment: bool=False) -> None:
        self._params = np.load(im_file)
        if augment:
            translate = transforms.RandomAffine(degrees=0, translate=(1/150, 1/30))
            _copy_params = np.copy(self._params) 
            _copy_values = np.copy(self._values) 
            _copy_preds = np.copy(self._preds) 
            for _ in range(4):
                aug = translate(torch.from_numpy(self._params).to(torch.float32))
                _copy_params = np.concatenate((_copy_params, aug.cpu().detach().numpy()), axis=0)
                _copy_values = np.concatenate((_copy_values, self._values), axis=0)
                _copy_preds = np.concatenate((_copy_preds, self._preds), axis=0)
            self._params = _copy_params
            self._values = _copy_values
            self._preds = _copy_preds

    def upsample(self):
        added_vals = []
        added_preds = []
        added_params = []
        for i in range(len(self.values)):
            upsample=0
            if self.values[i] < 2.2:
                upsample = 9
            elif self.values[i] < 3.2:
                upsample = 6
            elif self.values[i] < 4.2:
                upsample = 3
            for j in range(upsample):
                added_vals.append(self._values[i])
                added_preds.append(self._preds[i])
                added_params.append(self._params[i])
        self._values = np.concatenate((self._values, np.array(added_vals)))
        self._preds = np.concatenate((self._preds, np.array(added_preds)))
        self._params = np.concatenate((self._params, np.array(added_params)))

    def __getitem__(self, i) -> Any:
        return (torch.from_numpy(self._params[i]).to(torch.float32),
                torch.tensor(self._values[i]).to(torch.float32),
                torch.from_numpy(self._preds[i]).to(torch.float32),
                torch.from_numpy(self.pred_weights[i]).to(torch.float32))

    def __len__(self) -> int:
        return len(self._params)

