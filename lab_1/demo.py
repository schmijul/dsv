import avgSquareError as mse
from signal_src import show_spectrum_log, play, generate_signal
from scipy import signal

if __name__ == '__main__':
    x1 = generate_signal(1) #u(nT)
    x2 = generate_signal(2)
    x3 = generate_signal(3)
    x4 = generate_signal(4) #x(nT)
    

    #play(x1)
    #play(x2)
    #play(x3)
    #play(x4)
    
    
    #MSE durch das Rauschen
    #print(mse(x1, x2))
    #print(mse(x1, x3))
    #print(mse(x1, x4))
    
    #Filter Entwurf
    fs = 8000     # sampling freq
    fD = 600      # passband freq
    fS = 900      # stopband freq
    ripple = 18   # max difference to desired freq in db
    M = len(x1)   # number of points
    
    numtaps, beta = signal.kaiserord(ripple, 2*(fS-fD)/fs)
    
    kaiserfilter = (signal.firwin(numtaps, fD * 0.9, window = ('kaiser', beta), fs = fs))
    print(f'Filterordnung: {len(kaiserfilter)}')
    x_filt=signal.lfilter(kaiserfilter,1,x4)
    play(x_filt)
    show_spectrum_log([x_filt, x1])
    print(f'MSE: {mse.avgSquareError(x_filt,x1)}')
    print(f'MSE: {mse.avgSquareError(x4,x1)}')
    
