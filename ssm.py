from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy

class ssm:
    def __init__(self, audio_path, k = 10, t = 'mfcc', normalized = 1, smooth = 1, thresh = 1):
        self.audio, self.sr = self.read_audio(audio_path) # audio, sampling rate
        self.s = self.create_ssm(self.calculate_feat(t), normalized) # self similarity matrix
        self.reduce_ssm(k) # reduce the size of the ssm
        self.duration = self.duration() # duration of the audio

        if smooth == 1:
            self.path_smooth()
        if thresh == 1:
            self.threshold()

    def read_audio(self, audio_path):
        audio, sr = librosa.load(audio_path)
        return audio, sr

    def calculate_feat(self, t = 'chroma'):
        print("Calculating features...")
        if t == 'chroma':
            return librosa.feature.chroma_stft(y = self.audio, sr = self.sr, n_fft = 2048)
        elif t == 'tempo':
            oenv = librosa.onset.onset_strength(y = self.audio, sr = self.sr)
            feature = librosa.feature.tempogram(onset_envelope = oenv, sr = self.sr)
            return feature
        elif t == 'mfcc':
            mfccs = librosa.feature.mfcc(y = self.audio, sr = self.sr, n_mfcc=45)
            mfccs = mfccs[np.argsort(np.var(mfccs, axis = 1))[::-1]]
            mfccs = mfccs[:15]
            mfccs = (mfccs - np.mean(mfccs, axis = 1)[:, np.newaxis]) / np.std(mfccs, axis = 1)[:, np.newaxis]
            return mfccs

    def create_ssm(self, feat, normalized):
        print("Features calculated.")
        if normalized == 1:
            s_norm = np.linalg.norm(feat, axis = 0)
            s_norm[s_norm == 0] = 1
            feat   = np.abs(feat/s_norm)
        print("Calculating SSM...")
        s = np.dot(feat.T, feat)
        print("SSM calculated.")
        return s

    def create_ssm_old(self, feat):
        M, N = feat.shape
        s = np.zeros(N * N).reshape(N, N)
        for i in range(N):
            for j in range(N):
                s[i, j] = self.dist(feat[:, i], feat[:, j])
        return s

    def dist(self, f, g):
        return np.dot(f, g)

    def score(self, m, n):
        return self.s(m, n)
    
    def visualize_mfcc(self):
        import librosa.display
        feat = self.calculate_feat('mfcc')
        fig, ax = plt.subplots(figsize=(12, 3))   
        librosa.display.specshow(librosa.power_to_db(feat, ref=np.max), x_axis='s')
        ax.set_ylabel('MFCC')
        plt.set_cmap('viridis')
        plt.colorbar()
        plt.show()

    def visualize(self):
        import librosa.display
        fig, ax = plt.subplots(figsize=(12, 8))   
        librosa.display.specshow(self.s.T, x_axis='s', y_axis='s', sr=self.sr, win_length=2048, hop_length=2*2205)
        ax.set_title('SSM')
        plt.set_cmap('hot')
        plt.colorbar()
        ax.invert_yaxis()
        plt.show()

    def checkerboard_kernel(self, k):
        C = np.array([[1, -1], [-1, 1]])
        O = np.ones((k, k))
        return np.kron(C, O)
    
    def cross_corr(self, k):
        kern = self.checkerboard_kernel(k)
        return np.maximum(0, scipy.signal.correlate(self.s, kern, mode='same'))

    def visualize_cross_corr(self, k=2, d=None, pen=10):
        import ruptures as rpt
        fig, ax = plt.subplots(figsize=(12, 3))   
        ax.set_ylabel('Novelty score')
        ax.set_xlabel('Time (s)')
        cross_corr = self.cross_corr(k).sum(axis = 0)
        algo = rpt.Pelt(model="rbf").fit(cross_corr)
        result = algo.predict(pen=pen)
        for i in range(len(result) - 1):
            ax.axvspan(0, result[0] * self.duration / self.s.shape[0], color='lightblue', alpha=0.4)
            if i % 2 == 0:
                ax.axvspan(result[i] * self.duration / self.s.shape[0], result[i+1] * self.duration / self.s.shape[0], color='lightcoral', alpha=0.4)
            else:
                ax.axvspan(result[i] * self.duration / self.s.shape[0], result[i+1] * self.duration / self.s.shape[0], color='lightblue', alpha=0.4)
            ax.axvline(x = result[i] * self.duration / self.s.shape[0], linestyle='--', color = 'black')
        ax.plot(np.arange(self.s.shape[0]) * self.duration / self.s.shape[0], cross_corr)
        if d is not None:
            for dx in d:
                ax.axvline(x = dx, linestyle='--', color = 'r')
        plt.show()

    def visualize_img(self):
        S = Image.fromarray(self.s * 100)
        S.show()

    def duration(self):
        return librosa.core.get_duration(y=self.audio, sr=self.sr)

    def reduce_ssm(self, k):
        self.s = self.s[::k,::k]

    def threshold(self, tau = 0.8):
        self.s[self.s < tau] = 0

    def path_smooth(self, k = 20):
        self.s = scipy.ndimage.filters.median_filter(self.s,footprint = np.eye(k))
