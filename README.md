# METRO: Multi-channel ExTension of pRe-trained mOdel implementation based on WeSpeaker

This repository currently provides an implementation of a multi-channel extension of WavLM Base+. Please refer to `examples/multisv` for an example with the MultiSV dataset.

### Installation
* Clone this repo
``` sh
git clone https://github.com/BUTSpeechFIT/Wespeaker_MC_SSL.git
```

* Create conda env: pytorch version >= 1.10.0 is required
``` sh
conda create -n wespeaker python=3.9
conda activate wespeaker
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### The repository implements the following paper
@inproceedings{metro,
  author={Ladislav Mošner and Romain Serizel and Lukáš Burget and Oldřich Plchot and Emmanuel Vincent and Junyi Peng and Jan Černocký},
  title={{Multi-Channel Extension of Pre-trained Models for Speaker Verification}},
  year=2024,
  booktitle={Proc. Interspeech}
}
