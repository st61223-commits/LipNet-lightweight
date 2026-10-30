"""
溫度實驗：v11（.h5）與現役 α=0.5（ONNX）在「即時系統節奏」下的 CPU 溫度、時脈、功耗與延遲。
- 兩個模型都固定每 0.6 秒推論一次（對應 SLIDE_STEP=15、25 fps）；推論比 0.6 秒久時就接著跑下一次（v11 會一直滿載）
- 順序交替 v11 → α0.5 → v11 → α0.5（--reps 決定輪數），每輪 --minutes 分鐘
- 每輪開始前先等 CPU 冷卻到「起始待機溫度 + 2°C」以內（最多等 --max-cool 分鐘），讓每輪起點相同
- 感測器由 LibreHardwareMonitor 的 Remote Web Server（http://localhost:8085/data.json）每秒讀一次
用法：python thermal_experiment.py --check（只測試能否讀到感測器）
      python thermal_experiment.py [--minutes 15] [--reps 2]
輸出：LipNet/logs/thermal_sensors.csv、thermal_inference.csv、thermal_log.txt
"""
import os, sys, io, csv, json, time, argparse, threading, urllib.request
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 只用 CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

BASE = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
LOG_DIR = os.path.join(BASE, 'logs')
URL = 'http://localhost:8085/data.json'
PERIOD = 0.6
MODELS = {
    'v11': os.path.join(BASE, 'models', 'full_v11', 'trained_model_full_a100_v8.h5'),
    'a050_onnx': os.path.join(BASE, 'models', 'width05_yolov8.onnx'),
}


def walk(node, path=()):
    """攤平 LibreHardwareMonitor 的樹狀 JSON，回傳 (路徑, 名稱, 數值字串)。"""
    here = path + (node.get('Text', ''),)
    if node.get('Value') not in (None, ''):
        yield here, node.get('Text', ''), node['Value']
    for c in node.get('Children', []):
        yield from walk(c, here)


def num(v):
    try:
        return float(str(v).split()[0].replace(',', '.'))
    except ValueError:
        return float('nan')


def read_sensors():
    with urllib.request.urlopen(URL, timeout=1) as r:
        tree = json.load(r)
    temp = power = load = float('nan')
    clocks = []
    for path, name, val in walk(tree):
        group = path[-2] if len(path) >= 2 else ''
        in_cpu = any(('Intel' in p or 'AMD' in p or 'CPU' in p) for p in path[:-2])
        if not in_cpu:
            continue
        if group == 'Temperatures' and name == 'CPU Package':
            temp = num(val)
        elif group == 'Powers' and name == 'CPU Package':
            power = num(val)
        elif group == 'Clocks' and name.startswith(('CPU Core #', 'Core #', 'P-Core #', 'E-Core #')):
            clocks.append(num(val))
        elif group == 'Load' and name == 'CPU Total':
            load = num(val)
    clock = sum(clocks) / len(clocks) if clocks else float('nan')
    return temp, clock, power, load


class Sampler(threading.Thread):
    def __init__(self, writer, fp):
        super().__init__(daemon=True)
        self.writer, self.fp = writer, fp
        self.phase = 'idle'
        self.t0 = time.time()
        self.stop_flag = False
        self.last_temp = float('nan')
        self.fail_since = None

    def run(self):
        while not self.stop_flag:
            t = time.time()
            try:
                temp, clock, power, load = read_sensors()
            except Exception:
                temp = clock = power = load = float('nan')
            self.last_temp = temp
            self.fail_since = None if temp == temp else (self.fail_since or t)
            self.writer.writerow([f'{t - self.t0:.1f}', self.phase, temp, clock, power, load])
            self.fp.flush()
            time.sleep(max(0.0, 1.0 - (time.time() - t)))


def load_runner(name):
    import numpy as np
    x = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)
    if name.endswith('onnx'):
        import onnxruntime as ort
        sess = ort.InferenceSession(MODELS[name], providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0].name
        return lambda: sess.run(None, {inp: x})
    import tensorflow as tf
    m = tf.keras.models.load_model(MODELS[name], compile=False)
    return lambda: m(x, training=False).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--minutes', type=float, default=15)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--max-cool', type=float, default=10)
    args = ap.parse_args()

    # 2026-10-30：第一次正式跑時 LibreHardwareMonitor 在開始約 1 分鐘後關掉，之後全部讀不到；
    # 改成先等感測器可讀才開始（最多等 3 小時），量測中斷線超過 30 秒就中止，避免白跑
    t_wait = time.time()
    while True:
        try:
            temp, clock, power, load = read_sensors()
            break
        except Exception as e:
            if args.check or time.time() - t_wait > 3 * 3600:
                print(f'[錯誤] 讀不到 {URL}：{e}\n請確認 LibreHardwareMonitor 以系統管理員身分執行，並已開啟 Options → Remote Web Server → Run')
                sys.exit(1)
            time.sleep(5)
    print(f'感測器：CPU 溫度 {temp}°C、平均時脈 {clock:.0f} MHz、功耗 {power} W、使用率 {load}%')
    if args.check:
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    sfp = open(os.path.join(LOG_DIR, 'thermal_sensors.csv'), 'w', newline='', encoding='utf-8')
    sw = csv.writer(sfp); sw.writerow(['t', 'phase', 'temp_c', 'clock_mhz', 'power_w', 'load_pct'])
    ifp = open(os.path.join(LOG_DIR, 'thermal_inference.csv'), 'w', newline='', encoding='utf-8')
    iw = csv.writer(ifp); iw.writerow(['t', 'phase', 'latency_ms'])
    log = open(os.path.join(LOG_DIR, 'thermal_log.txt'), 'w', encoding='utf-8')

    def say(msg):
        print(msg); log.write(msg + '\n'); log.flush()

    sampler = Sampler(sw, sfp); sampler.start()

    def check_alive():
        if sampler.fail_since and time.time() - sampler.fail_since > 30:
            say('[中止] 感測器已超過 30 秒讀不到（LibreHardwareMonitor 可能被關掉），本次數據不完整')
            os._exit(2)
    say('先載入兩個模型並暖機（不計入量測）...')
    runners = {n: load_runner(n) for n in MODELS}
    for f in runners.values():
        for _ in range(3): f()

    say('待機 2 分鐘量測起始溫度...')
    sampler.phase = 'baseline'
    temps = []
    end = time.time() + 120
    while time.time() < end:
        time.sleep(1); temps.append(sampler.last_temp); check_alive()
    temps = [t for t in temps if t == t]
    base = sum(temps[-30:]) / len(temps[-30:])
    say(f'起始待機溫度 {base:.1f}°C')

    order = [n for _ in range(args.reps) for n in ('v11', 'a050_onnx')]
    for i, name in enumerate(order, 1):
        phase = f'{name}_r{(i + 1) // 2}'
        sampler.phase = f'cool_before_{phase}'
        t_cool = time.time()
        while sampler.last_temp > base + 2 and time.time() - t_cool < args.max_cool * 60:
            time.sleep(1); check_alive()
        check_alive()
        say(f'[{i}/{len(order)}] {phase} 開始（冷卻 {time.time() - t_cool:.0f} 秒，目前 {sampler.last_temp}°C）')
        sampler.phase = phase
        f = runners[name]
        t_end = time.time() + args.minutes * 60
        nxt = time.time()
        n = 0
        while time.time() < t_end:
            now = time.time()
            if now < nxt:
                time.sleep(nxt - now)
            t0 = time.time()
            f()
            lat = (time.time() - t0) * 1000
            iw.writerow([f'{t0 - sampler.t0:.2f}', phase, f'{lat:.1f}']); n += 1
            check_alive()
            nxt = max(nxt + PERIOD, time.time())
        ifp.flush()
        say(f'    {phase} 結束：共推論 {n} 次，結束時 {sampler.last_temp}°C')

    sampler.phase = 'cooldown_end'
    time.sleep(5)
    sampler.stop_flag = True; sampler.join()
    for fp in (sfp, ifp, log):
        fp.close()
    print('ALL DONE')


if __name__ == '__main__':
    main()
