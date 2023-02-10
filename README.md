# Audio Music Summarization via Similarity Analysis

Repository for the **Audio Signal Processing** MVA course project.

## Getting started

These instructions will guide you through the process of cloning the repository, installing the required packages and executing the code for creating a thumbnail of any song.

### Prerequisites

Before you start, make sure you have the following packages installed:

* numpy
* librosa

You can install these packages by running the following command:

```sh
pip install numpy librosa
```

### Cloning the repository

To clone the repository, run the following command in your terminal (for SSH):

```sh
git clone git@github.com:Exion35/audio-thumbnailing.git
```

### Running the code

After you have cloned the repository and installed the required packages, navigate to the repository diretory and execute the following code

```python
from cooper_foote import audio_thumb_cf
from IPython.display import Audio, display

PATH = '/path/to/file'
song = audio_thumb_cf(PATH)

ssm = song.ssm  # create similarity matrix
ssm.visualize() # visualize similarity matrix

length = 20 # length of the thumbnail
song.display_excerpt(length) # display the thumbnail
```

## Reference

* Cooper, Matthew & Foote, Jonathan. (2002). Automatic Music Summarization via Similarity Analysis. 