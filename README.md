# Audio Music Summarization via Similarity Analysis

Repository for the **Audio Signal Processing** MVA course project.

## Getting started

These instructions will guide you through the process of cloning the repository, installing the required packages and executing the code for creating a thumbnail of any song.

### Prerequisites

Before you start, make sure you have the following packages installed:

* `numpy`
* `scipy`
* `pydub`
* `librosa`
* `ruptures`

You can install these packages by running the following command:

```
pip install numpy scipy pydub librosa ruptures
```

### Cloning the repository

To clone the repository, run the following command in your terminal (for SSH):

```
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
ssm.visualize_mfcc() # visualize mfcc
ssm.visualize_cross_corr() # visualize novelty score

length = 20 # length of the thumbnail
song.display_excerpt(length) # display the thumbnail
```

## References

* Cooper, Matthew & Foote, Jonathan. (2002). Automatic Music Summarization via Similarity Analysis. 
* Jonathan Foote. Automatic audio segmentation using a measure of audio novelty. In 2000 ieee international conference on multimedia and expo. icme2000. proceedings. latest ad-
vances in the fast changing world of multimedia (cat. no. 00th8532), volume 1, pages 452–455.
IEEE, 2000
* Charles Truong, Laurent Oudre, and Nicolas Vayatis. Selective review of offline change
point detection methods. Signal Processing, 167:107299, 2020.  
[Ruptures package](https://github.com/deepcharles/ruptures)