# Audio Thumbnailing based on "Automatic Music Summarization via Similatiry Analysis",
# by Matthew Cooper and Jonathan Foote.

import numpy as np
import librosa
from IPython.display import Audio, display
from ssm import ssm
from pydub import AudioSegment
from scipy.io import wavfile

class audio_thumb_cf:
    def __init__(self, audio_path, format, t = 'chroma', k = 10, smooth = 1, thresh = 1):
        if format != 'wav':
            sound = AudioSegment.from_file(audio_path, format=format)
            y = np.array(sound.get_array_of_samples())
            sr = sound.frame_rate
            audio_path = audio_path.split('.')[0] + '.wav'
            wavfile.write(audio_path, 2*sr, y)
        self.ssm = ssm(audio_path, k, t, smooth, thresh)
        self.y, self.sr  = librosa.load(audio_path)
        self.time = 0

    def score_normalized(self, q, r):
        return np.sum(np.sum(self.ssm.s[:,q : r + 1], axis = 0)/(self.ssm.s.shape[0]*(r - q + 1)))

    def score_Q(self, L, i):
        return self.score_normalized(i, i + L - 1)

    def score_max(self, L):
        N = self.ssm.s.shape[0]
        s = np.array([])

        for i in range(N - L + 1):
            s = np.append(s, self.score_Q(L, i))
        return s.argmax()

    def thumb_alpha(self, L):
        q_max = self.score_max(L)
#        self.thumb_frame = q_max
        print("The best thumbnail for this song with length " + str(round(self.frame_to_time(L), 2)) + " starts at time: " + str(round(self.frame_to_time(q_max), 2)) + "s")


    def thumb_time(self, time):
        L = self.time_to_frame(time)
        q_max = self.score_max(L)
        self.time = self.frame_to_time(q_max)
        print("The best thumbnail for this song with length " + str(round(self.frame_to_time(L), 2)) + " starts at time: " + str(round(self.frame_to_time(q_max), 2)) + "s")
        return round(self.frame_to_time(L), 2), round(self.frame_to_time(q_max), 2)

    def frame_to_time(self, f):
        dt = self.ssm.duration/self.ssm.s.shape[0]
        return dt*f

    def time_to_frame(self, time):
        df = self.ssm.s.shape[0]/self.ssm.duration
        return int(df*time)

    def display_excerpt(self, length):
        """
        Displays the best thumbnail for the song for a given length.
        """
        d, start = self.thumb_time(length)
        display(Audio(self.y[int(start*self.sr) : int((start + d)*self.sr)], rate = self.ssm.sr))