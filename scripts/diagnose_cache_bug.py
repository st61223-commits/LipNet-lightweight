import os, glob
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
import numpy as np

def load_frames_raw(project_name, file_name):
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{project_name}{suffix}', f'{file_name}.npy')
        if os.path.exists(cache_path):
            return np.load(cache_path), cache_path
    return None, None

def load_align_raw(project_name, file_name):
    align_path = os.path.join('data', 'alignments', project_name.replace('_new', ''), f'{file_name}.align')
    if not os.path.exists(align_path):
        return None, align_path
    with open(align_path, 'r') as f:
        lines = f.readlines()
    return lines, align_path

SPEAKERS = ['s2', 's6', 's7']

for spk in SPEAKERS:
    print(f'=== {spk} ===', flush=True)
    pattern = os.path.join('data', spk, '*.mpg')
    files = sorted(glob.glob(pattern))
    print(f'  共 {len(files)} 支影片', flush=True)
    bad = []
    for fp in files:
        file_name = os.path.splitext(os.path.basename(fp))[0]
        frames, cache_path = load_frames_raw(spk, file_name)
        lines, align_path = load_align_raw(spk, file_name)
        problem = None
        if frames is None:
            problem = f'找不到快取檔'
        elif frames.ndim != 4 or frames.shape[0] != 75:
            problem = f'frames shape異常: {frames.shape}'
        elif np.isnan(frames).any():
            problem = 'frames含NaN'
        if lines is None:
            problem = (problem + '; ' if problem else '') + '找不到align檔'
        else:
            words = [l.split()[2] for l in lines if len(l.split()) >= 3]
            real_words = [w for w in words if w not in ('sil', 'sp') and len(w) > 1]
            if len(real_words) == 0:
                problem = (problem + '; ' if problem else '') + f'align無有效字詞(共{len(lines)}行)'
        if problem:
            bad.append((file_name, problem, frames.shape if frames is not None else None))
    print(f'  異常筆數: {len(bad)}', flush=True)
    for name, problem, shape in bad:
        print(f'    {name}: {problem}', flush=True)
