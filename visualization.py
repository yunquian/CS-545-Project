import numpy as np
import matplotlib.pyplot as plt


def plot_spec(amp, time=None, freq=None, title='',
              new_plot=True, figure_size=(10, 6)):
    """
    Draws spectrum
    :param amp: 2D numpy array of shape (freq, time)
    :param time: 1D numpy array of actual time
    :param freq: 1D numpy array of actual frequency
    :param title: Title of the plot
    :param new_plot: whether draws on new figure
    :param figure_size: (width, height) tuple
    """
    if new_plot:
        plt.figure(figsize=figure_size)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
    if time is None:
        time = np.arange(amp.shape[1])
    if freq is None:
        freq = np.arange(amp.shape[0])
    plt.pcolormesh(time, freq, amp, shading='auto', cmap=plt.cm.plasma)
    plt.colorbar()
