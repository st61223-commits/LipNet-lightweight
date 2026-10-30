"""
畫溫度實驗結果（資料來自 thermal_experiment.py）：
  溫度實驗_熱度圖.png   橫軸=每輪開始後的時間、縱軸=各輪（v11 / α=0.5 交替）、顏色=CPU 溫度
  溫度實驗_折線圖.png   溫度、推論延遲隨時間變化（時脈圖已依使用者要求拿掉）
  溫度實驗_長條圖.png   平均溫度、平均功耗、每次推論耗電
輸出到 OneDrive\桌面\專題\，摘要文字存 LipNet/logs/thermal_summary.txt
"""
import os, sys, io, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

LOG = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\logs'
OUT = r'C:\Users\Tno\OneDrive\桌面\專題'
LABEL = {'v11': 'v11（.h5）', 'a050_onnx': '現役模型（α = 0.5，ONNX）'}
COLOR = {'v11': '#2a78d6', 'a050_onnx': '#eb6834'}
BIN = 10  # 熱度圖每格 10 秒


def read(name):
    with open(os.path.join(LOG, name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


sens = read('thermal_sensors.csv')
infer = read('thermal_inference.csv')
runs = []
for r in sens:
    p = r['phase']
    if (p.startswith('v11_r') or p.startswith('a050_onnx_r')) and p not in runs:
        runs.append(p)


def model_of(run):
    return run.rsplit('_r', 1)[0]


def series(run, key):
    rows = [r for r in sens if r['phase'] == run]
    t0 = float(rows[0]['t'])
    t = np.array([float(r['t']) - t0 for r in rows])
    v = np.array([float(r[key]) for r in rows])
    return t, v


def latency(run):
    rows = [r for r in infer if r['phase'] == run]
    t0 = float(rows[0]['t'])
    return np.array([float(r['t']) - t0 for r in rows]), np.array([float(r['latency_ms']) for r in rows])


# ---------- 熱度圖 ----------
dur = min(series(r, 'temp_c')[0][-1] for r in runs)
nb = int(dur // BIN)
grid = np.full((len(runs), nb), np.nan)
for i, run in enumerate(runs):
    t, v = series(run, 'temp_c')
    for b in range(nb):
        m = (t >= b * BIN) & (t < (b + 1) * BIN)
        if m.any():
            grid[i, b] = np.nanmean(v[m])
fig, ax = plt.subplots(figsize=(11, 0.7 * len(runs) + 1.6))
im = ax.imshow(grid, aspect='auto', cmap='inferno', extent=[0, nb * BIN / 60, len(runs) - 0.5, -0.5])
ax.set_yticks(range(len(runs)))
ax.set_yticklabels([f'{LABEL[model_of(r)]}　第 {r.rsplit("_r", 1)[1]} 輪' for r in runs])
ax.set_xlabel('每輪開始後的時間（分鐘）')
ax.set_title('CPU 溫度熱度圖（每 0.6 秒推論一次，純 CPU）')
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label('CPU Package 溫度（°C）')
fig.tight_layout()
fig.savefig(os.path.join(OUT, '溫度實驗_熱度圖.png'), dpi=200)
plt.close(fig)

# ---------- 折線圖 ----------
def smooth(v, w=30):
    v = np.where(np.isnan(v), np.nanmean(v), v)
    p = np.pad(v, (w // 2, w - 1 - w // 2), mode='edge')  # 頭尾用邊界值補，避免平均時被 0 拉低
    return np.convolve(p, np.ones(w) / w, mode='valid')


# 使用者要求：只放第 1 輪（第 2 輪結果一致，熱度圖與長條圖仍含兩輪）；溫度與延遲拆成兩張獨立的圖；圖例放在圖外上方
first = [r for r in runs if r.endswith('_r1')]


def line_chart(filename, draw, ylabel):
    fig, ax = plt.subplots(figsize=(10, 3.9))
    for run in first:
        draw(ax, run, model_of(run))
    ax.set_ylabel(ylabel)
    ax.set_xlabel('開始後的時間（分鐘）')
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.savefig(os.path.join(OUT, filename), dpi=200)
    plt.close(fig)


def draw_temp(ax, run, m):
    # 感測器每秒一筆、上下跳動很大，改畫 30 秒移動平均，趨勢才看得清楚
    t, v = series(run, 'temp_c')
    ax.plot(t / 60, smooth(v), color=COLOR[m], label=LABEL[m], lw=1.6)


def draw_latency(ax, run, m):
    t, v = latency(run)
    ax.plot(t / 60, v, color=COLOR[m], label=LABEL[m], lw=0.8)
    if m == 'v11':
        ax.axhline(600, color='gray', lw=0.8, ls=':')
        ax.text(0.1, 620, '0.6 秒推論間隔', color='gray', fontsize=9)


line_chart('溫度實驗_溫度折線圖.png', draw_temp, 'CPU 溫度（°C，30 秒平均）')
line_chart('溫度實驗_延遲折線圖.png', draw_latency, '單次推論延遲（ms）')

# ---------- 長條圖 ----------
stats = {}
for m in LABEL:
    temps, powers, energy, n_inf, lats = [], [], 0.0, 0, []
    for run in [r for r in runs if model_of(r) == m]:
        t, v = series(run, 'temp_c'); temps += list(v[t >= t[-1] - 300])  # 最後 5 分鐘（接近穩定）
        t, p = series(run, 'power_w'); powers += list(p); energy += np.nansum(p) * 1.0  # 每秒一筆 → 焦耳
        _, l = latency(run); n_inf += len(l); lats += list(l)
    stats[m] = {'temp': np.nanmean(temps), 'power': np.nanmean(powers), 'joule': energy / n_inf,
                'lat': np.mean(lats), 'n': n_inf}
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
keys = [('temp', '穩定後平均溫度（°C）'), ('power', '平均 CPU 功耗（W）'), ('joule', '每次推論耗電（J）')]
for a, (k, title) in zip(axes, keys):
    vals = [stats[m][k] for m in LABEL]
    bars = a.bar([LABEL[m] for m in LABEL], vals, color=[COLOR[m] for m in LABEL], width=0.55)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}', ha='center', va='bottom')
    a.set_title(title); a.grid(axis='y', alpha=0.3); a.tick_params(axis='x', labelsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '溫度實驗_長條圖.png'), dpi=200)
plt.close(fig)

lines = [f'{LABEL[m]}：穩定後平均溫度 {s["temp"]:.1f}°C、平均功耗 {s["power"]:.1f} W、每次推論耗電 {s["joule"]:.2f} J、'
         f'平均延遲 {s["lat"]:.1f} ms、推論 {s["n"]} 次' for m, s in stats.items()]
with open(os.path.join(LOG, 'thermal_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
