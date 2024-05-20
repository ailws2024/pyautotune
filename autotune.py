import librosa
import soundfile as sf
import pyworld as pw
import numpy as np
from scipy.signal import savgol_filter

# Function to shift pitch using librosa
def shift_pitch(y, sr, semitones):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

# Load the audio file
y, sr = librosa.load('your_file_and_file_extension', sr=None)

# Define a C major scale in Hz
C_MAJOR_SCALE = [32.70, 36.71, 41.20, 43.65, 49.00, 55.00, 61.74, 
                 65.41, 73.42, 82.41, 87.31, 98.00, 110.00, 123.47, 
                 130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 
                 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 
                 440.00, 493.88, 523.25, 587.33, 659.26, 698.46, 
                 783.99, 880.00, 987.77, 1046.50]

# Function to find closest note frequency in the scale
def closest_note_frequency(frequency):
    index = (np.abs(np.array(C_MAJOR_SCALE) - frequency)).argmin()
    return C_MAJOR_SCALE[index]

# Convert the buffer to double (64-bit float) as required by pyworld
y = y.astype(np.float64)

# Extract the pitch (fundamental frequency) and harmonic structure
f0, time = pw.harvest(y, sr)

# Smooth the pitch contour
f0_smooth = savgol_filter(f0, 51, 3)  # Window size 51, polynomial order 3

# Correct the pitch to the nearest note in the scale
f0_corrected = np.array([closest_note_frequency(p) if p > 0 else 0 for p in f0_smooth])

# Extract spectral envelope and aperiodicity
sp = pw.cheaptrick(y, f0_corrected, time, sr)
ap = pw.d4c(y, f0_corrected, time, sr)

# Synthesize the modified audio
y_tuned = pw.synthesize(f0_corrected, sp, ap, sr)

# Save the output to a new .wav file
sf.write('your_output_file', y_tuned, sr)
