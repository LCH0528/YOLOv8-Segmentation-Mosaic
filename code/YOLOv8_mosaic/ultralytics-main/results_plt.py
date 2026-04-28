from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def plot_results(file='results CHIP 1500 epochs.csv', dir=''):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob(file))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for fi, f in enumerate(files):
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            # 左半边：[1, 2, 3, 4, 13, 14, 15, 16]
            # 右半边：[5, 6, 9, 10, 7, 8, 11, 12]
            # 全部：[1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]
            for i, j in enumerate([1, 2, 3, 4, 13, 14, 15, 16]):
                y = data.values[:, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label="results", linewidth=2, markersize=8)
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)  # 修改保存路径
    plt.show()


if __name__ == '__main__':
    plot_results(file='results CHIP 1500 epochs.csv')  # 该python文件位于根目录下（此文件和传入文件在同一目录下），注意修改传入文件路径
