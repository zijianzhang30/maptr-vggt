"""Plot aggregate gradient conflict summaries."""
import argparse, csv, os
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('summary_csv'); ap.add_argument('--out-dir', default=None)
    a = ap.parse_args(); out = a.out_dir or os.path.dirname(a.summary_csv); rows = list(csv.DictReader(open(a.summary_csv)))
    rows.sort(key=lambda r: int(''.join(c for c in r['checkpoint'] if c.isdigit()) or 0))
    x = [int(''.join(c for c in r['checkpoint'] if c.isdigit()) or 0) for r in rows]
    for key, ylabel, filename, scale in [('mean_cosine','Mean gradient cosine','gradient_cosine_vs_epoch.png',1), ('conflict_rate','Conflict rate (%)','conflict_rate_vs_epoch.png',100), ('mean_norm_ratio','||g_VGGT|| / ||g_Map||','gradient_norm_ratio_vs_epoch.png',1)]:
        y = [float(r[key])*scale for r in rows]; plt.figure(); plt.plot(x,y,'o-');
        if key == 'mean_cosine': plt.axhline(0,color='k',ls='--',lw=.8)
        plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(out,filename),dpi=180); plt.close()
if __name__ == '__main__': main()
