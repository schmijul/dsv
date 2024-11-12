import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import copy

fs = 8000

def play(audio):
    a = copy.copy(audio)
    a *= (2**15 - 1) / np.max(np.abs(a))
    a = a.astype(np.int16)
    # Directly play audio using sounddevice
    sd.play(a, fs)
    sd.wait()  # Wait until audio playback is finished

def generate_sine(f, sec, add_hamming = True):
    num = round(sec * fs)
    t = np.linspace(0, sec, num, False)
    sine = np.sin(f * t * 2 * np.pi)
    if add_hamming:
        hamming = np.hamming(len(sine))
        sine *= hamming 
    return sine   
    
def add_noise(signal, Volume = 0.2):
    rand = np.random.randn(len(signal)) 
    return np.add(rand* Volume, signal)
    
def add_interference(signal, f, Volume = 0.3):
    sec = len(signal) / fs
    interference = generate_sine(f, sec, False)
    return np.add( interference* Volume, signal)
    
    
def create_plot(signal):
    plt.figure()
    spectrum = np.fft.fft(signal)/np.sqrt(len(signal))
    freq = np.fft.fftfreq(signal.size, 1/fs)
    spectrum = np.fft.fftshift(spectrum)
    freq = np.fft.fftshift(freq)
    spectrum = 20 * np.log10(np.absolute(spectrum))
    plt.plot(freq, spectrum)
    plt.ylabel('$|X(e^{j\omega T})|$ [dB]')
    plt.xlabel('$f$ [Hz]')
    
def show_spectrum_log(signal_):
    if isinstance(signal_, list):
        for signal in signal_:
            create_plot(signal)
    else:
        create_plot(signal_)
    plt.show()
            
    
def generate_signal(signal):
    A = generate_sine(440,0.25)
    C = generate_sine(440*2**(-9/12),0.25)
    D = generate_sine(440*2**(-7/12),0.25)
    E = generate_sine(440*2**(-5/12),0.25)
    F = generate_sine(440*2**(-4/12),0.25)
    G = generate_sine(440*2**(-2/12),0.25)
    H = generate_sine(440*2**(2/12),0.25)
    C2 = generate_sine(440*2**(3/12),0.25)
    

    tones = np.concatenate(((C,D,E,F,G,A,H,C2)))
    
    if (signal == 2) or (signal == 4):
        tones = add_noise(tones,0.2)
        
    if (signal == 3) or (signal == 4):
        tones = add_interference(tones, 950, 0.3)

    
    return(tones)
    
if __name__ == '__main__':
    signal = generate_signal(4)
    play(signal)
    plt.plot(signal)
    plt.show()
    show_spectrum_log(signal)
  
    
