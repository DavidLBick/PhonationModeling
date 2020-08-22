import os
from typing import Dict, List

import numpy as np
from scipy.io import wavfile
from textgrids import Interval, TextGrid

Tier = List[Interval]


def dict_phone_interval(tier: Tier) -> Dict[str, List[List[float]]]:
    """Create a dict mapping from phone name to time intervals.

    Args:
        tier: List[Interval]

    Returns:
        phone_dict: Dict[phone_name: str, List[List[t_min: float, t_max: float]]]
    """
    phone_dict = dict()
    for intvl in tier:
        if intvl.text not in phone_dict:
            phone_dict[intvl.text] = [[intvl.xmin, intvl.xmax]]
        else:
            phone_dict[intvl.text].append([intvl.xmin, intvl.xmax])
    return phone_dict


def get_phone_segments(
    tier: Tier, ph_intvls: List[List[float]], from_creaky: bool = True
) -> List[List[float]]:
    """Get time intervals for phone segments within a creaky tier.

    Args:
        tier: List[Interval]
            Tier containing creaky voices.
        ph_intvls: List[List[t_min: float, t_max: float]]
            Time intervals for a phone.
        from_creaky: bool, default True
            Indicates if extract from creaky intervel.

    Returns:
        ph_segs: List[List[t_min: float, t_max: float]]
            Time intervals for phone segments within a creaky tier.
    """
    ph_segs = []
    for intvl in tier:
        if (from_creaky) and (intvl.text == "c"):
            t_min, t_max = intvl.xmin, intvl.xmax
            for [pt_min, pt_max] in ph_intvls:
                if (t_min <= pt_min) and (pt_max <= t_max):
                    ph_segs.append([pt_min, pt_max])
                elif (pt_min < t_min) and (t_min < pt_max) and (pt_max <= t_max):
                    ph_segs.append([t_min, pt_max])
                elif (t_min <= pt_min) and (pt_min < t_max) and (t_max < pt_max):
                    ph_segs.append([pt_min, t_max])

        elif (not from_creaky) and (intvl.text != "c"):
            t_min, t_max = intvl.xmin, intvl.xmax
            for [pt_min, pt_max] in ph_intvls:
                if (t_min <= pt_min) and (pt_max <= t_max):
                    ph_segs.append([pt_min, pt_max])
                elif (pt_min < t_min) and (t_min < pt_max) and (pt_max <= t_max):
                    ph_segs.append([t_min, pt_max])
                elif (t_min <= pt_min) and (pt_min < t_max) and (t_max < pt_max):
                    ph_segs.append([pt_min, t_max])
    return ph_segs


def get_wav_segments(
    wav: np.ndarray, sample_rate: int, ph_segs: List[List[float]]
) -> List[np.ndarray]:
    """Get wav segments c.r.t. creaky phone segments.

    Args:
        wav: np.ndarray[np.float32]
            wav samples.
        sample_rate: int
        ph_segs: List[List[t_min: float, t_max: float]]

    Returns:
        wav_segs: List[np.ndarray]
            Wav segments of creaky phones.
    """

    def _t_2_index(ph_segs: List[List[float]]) -> List[List[int]]:
        """Convert time intervals to index intervals.

        Args:
            ph_segs: List[List[t_min: float, t_max: float]]

        Returns:
            ph_segs_idx: List[List[i_min: int, i_max: int]]
        """
        ph_segs_idx = [
            [int(np.ceil(pt_min * sample_rate)), int(np.floor(pt_max * sample_rate))]
            for [pt_min, pt_max] in ph_segs
        ]
        return ph_segs_idx

    ph_segs_idx = _t_2_index(ph_segs)
    wav_segs = [wav[i_min:i_max] for [i_min, i_max] in ph_segs_idx]
    return wav_segs


if __name__ == "__main__":
    data_root = ""
    project_root = ""
    save_dir = os.path.join(project_root, "")
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        print(f"folder {save_dir} already exists")

    wav_lst = [line.rstrip() for line in open("")]
    txtgrd_lst = [line.rstrip() for line in open("")]

    cnt = 1
    for wf, tf in zip(wav_lst, txtgrd_lst):
        print(f"Processing {wf}")

        # Read wav
        sample_rate, wav_raw = wavfile.read(os.path.join(data_root, "wavs", wf))

        # Convert from 16-bit int to 32-bit float
        wav_data = (wav_raw / pow(2, 15)).astype("float32")

        # Read textgrid
        txtgrd = TextGrid(os.path.join(data_root, tf))
        tier_phone = txtgrd[f"s{cnt} - phone"]  # tier containing phones
        tier_c = txtgrd["ipp"]  # tier containing creaky voices
        cnt = cnt + 1

        # Get creaky phone segments
        ph_intvls = dict_phone_interval(tier_phone)["AA1"]  # NOTE: phone
        ph_segs = get_phone_segments(tier_c, ph_intvls, from_creaky=False)
        wav_segs = get_wav_segments(wav_data, sample_rate, ph_segs)

        # Save to wav
        for i, w_seg in enumerate(wav_segs):
            wavfile.write(
                os.path.join(save_dir, wf.rstrip(".wav") + f"_phone_AA1_{i:d}.wav"),
                sample_rate,
                w_seg,
            )
        print(f"Done")
