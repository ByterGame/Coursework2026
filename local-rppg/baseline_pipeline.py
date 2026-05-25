import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import scipy.signal


HR_LOW_HZ = 0.7
HR_HIGH_HZ = 4.0


def _hann(n):
    """Hann window of length n."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))


def _next_pow2(n):
    """Smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _fft_radix2(x):
    """Cooley-Tukey radix-2 decimation-in-time FFT (in-place, recursive).

    Input length must be a power of 2. Returns complex array of same length.
    """
    N = len(x)
    if N <= 1:
        return x.astype(complex)
    even = _fft_radix2(x[0::2])
    odd = _fft_radix2(x[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([even + twiddle * odd,
                           even - twiddle * odd])


def _rfft(x, n=None):
    """Real FFT: compute the first N//2+1 bins via a full complex FFT.

    Zero-pads to `n` if given, otherwise to the next power of 2 >= len(x).
    """
    if n is None:
        n = _next_pow2(len(x))
    if n > len(x):
        x = np.pad(x, (0, n - len(x)))
    X = _fft_radix2(x)
    return X[:n // 2 + 1]


def _rfftfreq(n, d=1.0):
    """Frequencies for a real FFT of length n with sample spacing d."""
    return np.arange(n // 2 + 1) / (n * d)


def _welch_psd(sig, fs, nperseg=None, nfft=2048):
    """Welch PSD estimate — manual implementation.

    Splits `sig` into 50%-overlapping segments of length `nperseg`,
    applies a Hann window, computes |FFT|^2 per segment, and averages.
    Returns (freqs, psd) arrays.
    """
    N = len(sig)
    if nperseg is None:
        nperseg = min(N, int(fs * 8))
    nperseg = min(nperseg, N)
    nfft = _next_pow2(nfft)
    step = nperseg // 2  # 50 % overlap
    win = _hann(nperseg)
    win_power = np.sum(win ** 2)

    # collect segments
    psd_sum = np.zeros(nfft // 2 + 1)
    n_seg = 0
    start = 0
    while start + nperseg <= N:
        segment = sig[start:start + nperseg]
        windowed = (segment - segment.mean()) * win
        spectrum = np.abs(_rfft(windowed, n=nfft)) ** 2
        psd_sum += spectrum
        n_seg += 1
        start += step

    if n_seg == 0:
        return np.array([]), np.array([])

    psd = psd_sum / (n_seg * fs * win_power)
    # one-sided spectrum: double all bins except DC (0) and Nyquist (last)
    psd[1:-1] *= 2.0
    freqs = _rfftfreq(nfft, d=1.0 / fs)
    return freqs, psd


def forehead_mask(landmarks, h, w, frame=None):
    """Forehead ROI: bbox from face proportions, optionally narrowed to skin pixels.

    Step 1 (bbox): FaceMesh does not reach the hairline; we extrapolate upward
    from the brow line using a fixed proportion of the brow-to-chin distance.
    Step 2 (skin, if `frame` is given): YCrCb thresholding (Cr in [133, 173],
    Cb in [77, 127]) + morphological open/close drops hair, brows, deep shadows
    from the bbox.
    """
    xs = np.array([p.x for p in landmarks]) * w
    x_min, x_max = xs.min(), xs.max()
    fw = x_max - x_min

    y_brow = (landmarks[105].y + landmarks[334].y) / 2.0 * h
    y_chin = landmarks[152].y * h
    brow_to_chin = max(1.0, y_chin - y_brow)
    forehead_h = brow_to_chin * 0.55

    top_y = y_brow - forehead_h * 0.53
    bot_y = y_brow - forehead_h * 0.15

    cx = (x_min + x_max) / 2.0
    x_left = cx - fw * 0.36
    x_right = cx + fw * 0.36

    pts = np.array([
        [x_left, top_y], [x_right, top_y],
        [x_right, bot_y], [x_left, bot_y],
    ], dtype=np.int32)
    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(bbox_mask, [pts], 255)

    if frame is None:
        return bbox_mask, pts

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    final = cv2.bitwise_and(bbox_mask, skin)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, k)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, k)
    return final, pts

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "face_landmarker.task"


def ensure_model():
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"Downloading face landmark model -> {MODEL_PATH.name} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"  saved {MODEL_PATH.stat().st_size / 1024:.0f} KB")
    return MODEL_PATH


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--subject", type=Path, help="UBFC1 subject folder (vid-*.avi + gtdump.xmp)")
    src.add_argument("--video", type=Path, help="Plain video file (no ground truth)")
    p.add_argument("--method", choices=["green", "pos", "chrom"], default="pos",
                   help="Pulse extraction method (default: pos)")
    p.add_argument("--window", type=float, default=10.0, help="Sliding window length, s")
    p.add_argument("--step", type=float, default=1.0, help="Sliding window step, s")
    p.add_argument("--plot", action="store_true", help="Show diagnostic plots")
    p.add_argument("--save-plot", type=Path, default=None, help="Save plots to PNG (no GUI)")
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames (debug)")
    return p.parse_args()


def load_ubfc1_gt(gt_path):
    """gtdump.xmp: CSV with columns timestamp_ms, HR_bpm, SpO2, PPG_value.

    Returns dict with arrays t_s, hr_bpm, ppg, plus sample rate fs_gt.
    """
    data = np.loadtxt(gt_path, delimiter=",")
    t_ms = data[:, 0]
    hr = data[:, 1]
    ppg = data[:, 3]
    t_s = (t_ms - t_ms[0]) / 1000.0
    fs_gt = 1.0 / np.mean(np.diff(t_s))
    return {"t_s": t_s, "hr_bpm": hr, "ppg": ppg, "fs_gt": fs_gt}


def find_subject_files(subject_dir):
    subject_dir = Path(subject_dir)
    if not subject_dir.is_dir():
        sys.exit(f"--subject must be a directory: {subject_dir}")
    videos = sorted(subject_dir.glob("vid*.avi")) + sorted(subject_dir.glob("vid*.mp4"))
    gt = subject_dir / "gtdump.xmp"
    if not videos:
        sys.exit(f"No video file found in {subject_dir}")
    if not gt.exists():
        sys.exit(f"No gtdump.xmp found in {subject_dir}")
    return videos[0], gt


def extract_rgb_series(video_path, max_frames=None):
    """Run FaceLandmarker on every frame, return arrays of mean R, G, B over forehead mask."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n_frames = min(n_frames, max_frames)

    model_path = ensure_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    R, G, B = [], [], []
    miss = 0
    t0 = time.time()
    last_print = 0

    DOWNSCALE_MAX_SIDE = 1280

    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        # downsample huge frames (4K phone video) - speeds up mediapipe ~2x
        # without harming rPPG: the green-channel pulse modulation is the same
        # at any resolution as long as ROI stays stable across frames.
        h0, w0 = frame.shape[:2]
        scale = DOWNSCALE_MAX_SIDE / max(h0, w0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(idx / fps * 1000) if fps > 0 else idx
        res = landmarker.detect_for_video(mp_image, ts_ms)
        if not res.face_landmarks:
            R.append(np.nan); G.append(np.nan); B.append(np.nan); miss += 1
        else:
            lm = res.face_landmarks[0]
            h, w = frame.shape[:2]
            # bbox only: per-frame skin-mask hurts more than helps
            # (mask varies frame-to-frame -> noise > pulse modulation)
            mask, _ = forehead_mask(lm, h, w)
            m = cv2.mean(frame, mask=mask)  # B, G, R, _
            B.append(m[0]); G.append(m[1]); R.append(m[2])

        if idx - last_print >= max(1, n_frames // 20):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_frames - idx - 1) / rate if rate > 0 else 0
            print(f"  frame {idx+1}/{n_frames}  ({rate:.1f} fps, eta {eta:.0f}s)", end="\r")
            last_print = idx

    cap.release()
    landmarker.close()
    print()
    print(f"  done: {len(R)} frames, {miss} without face")

    R = np.array(R); G = np.array(G); B = np.array(B)
    for arr in (R, G, B):
        bad = np.isnan(arr)
        if bad.any():
            arr[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), arr[~bad])
    return R, G, B, fps


def pos_pulse_signal(R, G, B, fs, win_sec=1.6):
    """POS algorithm (Wang et al. 2017), plane-orthogonal-to-skin.

    Sliding window of length L = round(fs * win_sec):
      - normalize RGB by per-window mean
      - project with P = [[0, 1, -1], [-2, 1, 1]] to (S1, S2)
      - h = S1 + (sigma(S1)/sigma(S2)) * S2, mean-centered
      - overlap-add into the output buffer
    """
    rgb = np.stack([R, G, B], axis=0).astype(float)  # (3, N)
    N = rgb.shape[1]
    L = max(8, int(round(fs * win_sec)))
    H = np.zeros(N)
    if N < L:
        return H
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for n in range(L - 1, N):
        m = n - L + 1
        block = rgb[:, m:n + 1]
        means = block.mean(axis=1, keepdims=True)
        means[means == 0] = 1.0
        Cn = block / means
        S = P @ Cn  # (2, L)
        s1, s2 = S[0], S[1]
        sd2 = np.std(s2)
        alpha = np.std(s1) / sd2 if sd2 > 0 else 0.0
        h = s1 + alpha * s2
        h = h - h.mean()
        H[m:n + 1] += h
    return H


def chrom_pulse_signal(R, G, B, fs, win_sec=1.6):
    """CHROM algorithm (de Haan & Jeanne 2013).

    Sliding-window normalization -> Xs = 3R-2G, Ys = 1.5R+G-1.5B
    -> S = Xs - alpha*Ys, alpha = std(Xs)/std(Ys).
    """
    N = len(R)
    L = max(8, int(round(fs * win_sec)))
    S_out = np.zeros(N)
    if N < L:
        return S_out
    R = R.astype(float); G = G.astype(float); B = B.astype(float)
    for n in range(L - 1, N):
        m = n - L + 1
        r_win = R[m:n + 1]; g_win = G[m:n + 1]; b_win = B[m:n + 1]
        mr = r_win.mean(); mg = g_win.mean(); mb = b_win.mean()
        if mr == 0: mr = 1.0
        if mg == 0: mg = 1.0
        if mb == 0: mb = 1.0
        rn = r_win / mr; gn = g_win / mg; bn = b_win / mb
        xs = 3.0 * rn - 2.0 * gn
        ys = 1.5 * rn + gn - 1.5 * bn
        sd_ys = np.std(ys)
        alpha = np.std(xs) / sd_ys if sd_ys > 0 else 0.0
        h = xs - alpha * ys
        h = h - h.mean()
        S_out[m:n + 1] += h
    return S_out


def get_pulse_signal(R, G, B, fs, method):
    if method == "green":
        return G.astype(float)
    if method == "pos":
        return pos_pulse_signal(R, G, B, fs)
    if method == "chrom":
        return chrom_pulse_signal(R, G, B, fs)
    raise ValueError(f"Unknown method: {method}")


def estimate_hr(signal, fs, low=HR_LOW_HZ, high=HR_HIGH_HZ):
    """detrend -> Butterworth bandpass (filtfilt) -> Welch PSD -> peak in [low, high]."""
    if len(signal) < int(fs * 2):
        return np.nan, signal, np.array([]), np.array([])
    detrended = scipy.signal.detrend(signal)
    nyq = fs / 2.0
    b, a = scipy.signal.butter(3, [low / nyq, high / nyq], btype="band")
    filtered = scipy.signal.filtfilt(b, a, detrended)
    nperseg = min(len(filtered), int(fs * 8))
    f, pxx = _welch_psd(filtered, fs=fs, nperseg=nperseg, nfft=2048)
    band = (f >= low) & (f <= high)
    if not band.any():
        return np.nan, filtered, f, pxx
    peak_f = f[band][np.argmax(pxx[band])]
    return peak_f * 60.0, filtered, f, pxx


def sliding_window_eval(green, fs, gt, window_s, step_s):
    """Run estimate_hr on each window. Align with ground-truth HR(t)."""
    W = int(window_s * fs)
    S = int(step_s * fs)
    if W >= len(green):
        return np.array([]), np.array([]), np.array([])
    times, pred, ref = [], [], []
    for start in range(0, len(green) - W + 1, S):
        g_win = green[start:start + W]
        hr_pred, _, _, _ = estimate_hr(g_win, fs)
        t0 = start / fs
        t1 = (start + W) / fs
        if gt is not None:
            mask = (gt["t_s"] >= t0) & (gt["t_s"] <= t1)
            hr_gt = float(np.mean(gt["hr_bpm"][mask])) if mask.any() else np.nan
        else:
            hr_gt = np.nan
        times.append((t0 + t1) / 2.0)
        pred.append(hr_pred)
        ref.append(hr_gt)
    return np.array(times), np.array(pred), np.array(ref)


def make_dc_ac_plot(times_s, G, fs, save_path=None):
    """Show raw green channel: huge DC offset (skin/illumination) vs tiny AC (pulse)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [2, 1]})

    # Top: raw signal with DC
    ax1.plot(times_s, G, color="green", lw=0.5, alpha=0.8)
    ax1.set_title("Зелёный канал (область лба)")
    ax1.set_ylabel("значение пикселя (0–255)")
    ax1.set_xlabel("")
    ax1.grid(True, alpha=0.3)
    dc_val = np.mean(G)
    ax1.axhline(dc_val, color="red", ls="--", lw=1, label=f"DC среднее = {dc_val:.1f}")
    ax1.legend(fontsize=9)

    # Bottom: detrended (AC component only)
    G_detrended = G - np.mean(G)
    ax2.plot(times_s, G_detrended, color="green", lw=0.5, alpha=0.8)
    ax2.set_title("После удаления DC (выравнивание) — AC-компонента (пульс)")
    ax2.set_ylabel("отклонение от среднего")
    ax2.set_xlabel("время, с")
    ax2.grid(True, alpha=0.3)
    ac_amp = np.std(G_detrended)
    ax2.annotate(f"AC std = {ac_amp:.3f}", xy=(0.02, 0.92), xycoords="axes fraction",
                 fontsize=9, color="red")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=110)
        print(f"  plot saved -> {save_path}")
    else:
        plt.show()


def make_plots(times_s, pulse, method_name, fs, filtered, f, pxx, peak_hr_bpm,
               win_t, win_pred, win_ref, gt, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    band = (f >= HR_LOW_HZ) & (f <= HR_HIGH_HZ)
    ax.semilogy(f, pxx, color="grey", lw=0.6, label="PSD (Welch)")
    ax.semilogy(f[band], pxx[band], color="#1f77b4", lw=1.2, label="диапазон ЧСС")
    ax.axvline(peak_hr_bpm / 60.0, color="red", ls="--", lw=1, label=f"пик = {peak_hr_bpm:.1f} уд/мин")
    ax.set_xlim(0, 5); ax.set_xlabel("частота, Гц"); ax.set_ylabel("PSD")
    ax.set_title("Спектр Welch (весь сигнал)"); ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(win_t, win_pred, color="#d62728", lw=1.4, label=f"ЧСС предсказанная ({method_name})")
    if gt is not None and np.isfinite(win_ref).any():
        ax.plot(win_t, win_ref, color="black", lw=1.4, label="ЧСС эталонная")
    ax.set_xlabel("время, с"); ax.set_ylabel("ЧСС, уд/мин")
    ax.set_xlim(0, times_s[-1])
    ax.set_title(f"ЧСС по скользящему окну ({len(win_t)} окон)")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if gt is not None:
        T = times_s[-1]
        t_lo = max(0, T / 2 - 5); t_hi = min(T, T / 2 + 5)
        m_vid = (times_s >= t_lo) & (times_s <= t_hi)
        m_gt = (gt["t_s"] >= t_lo) & (gt["t_s"] <= t_hi)

        def norm(x):
            x = x - np.mean(x); s = np.std(x)
            return x / s if s > 0 else x
        ax.plot(times_s[m_vid], norm(filtered[m_vid]), color="#d62728",
                lw=1.2, label=f"{method_name} (наш фильтр)")
        ax.plot(gt["t_s"][m_gt], norm(gt["ppg"][m_gt]), color="black",
                lw=1.2, label="эталонный PPG")
        ax.set_title(f"Наложение сигналов, {t_lo:.1f}–{t_hi:.1f} с (z-нормализация)")
        ax.set_xlabel("время, с"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=110)
        print(f"  plot saved -> {save_path}")
    else:
        plt.show()


def main():
    args = parse_args()

    if args.subject:
        video_path, gt_path = find_subject_files(args.subject)
        gt = load_ubfc1_gt(gt_path)
        label = str(args.subject)
        print(f"Subject:        {label}")
        print(f"Ground truth:   {gt_path.name}, {len(gt['t_s'])} samples @ {gt['fs_gt']:.1f} Hz, "
              f"duration {gt['t_s'][-1]:.1f} s, mean HR {np.mean(gt['hr_bpm']):.1f} bpm")
    else:
        video_path = args.video
        gt = None
        label = str(args.video)
        print(f"Video:          {label}")

    print(f"Reading video:  {video_path.name}")
    R, G, B, fps = extract_rgb_series(video_path, max_frames=args.max_frames)
    n = len(G)
    duration = n / fps
    print(f"Frames:         {n} @ {fps:.2f} fps  ({duration:.1f} s)")

    if gt is not None and abs(duration - gt["t_s"][-1]) > 1.0:
        print(f"  WARNING: video duration {duration:.1f}s != gt duration {gt['t_s'][-1]:.1f}s")

    pulse = get_pulse_signal(R, G, B, fps, args.method)
    print(f"Method:         {args.method.upper()}")

    # Global estimate
    global_hr, filtered, f, pxx = estimate_hr(pulse, fps)
    print(f"\n=== Global HR ({args.method.upper()}) ===")
    print(f"  predicted:    {global_hr:.2f} bpm")
    if gt is not None:
        gt_mean = float(np.mean(gt["hr_bpm"]))
        print(f"  ground truth: {gt_mean:.2f} bpm  (mean of HR series)")
        print(f"  abs error:    {abs(global_hr - gt_mean):.2f} bpm")

    # Windowed estimate
    print(f"\n=== Sliding window (W={args.window}s, step={args.step}s) ===")
    win_t, win_pred, win_ref = sliding_window_eval(pulse, fps, gt, args.window, args.step)
    if len(win_t) == 0:
        print("  signal too short for the requested window")
    else:
        print(f"  {len(win_t)} windows, predicted HR: "
              f"mean {np.nanmean(win_pred):.1f} bpm, "
              f"std {np.nanstd(win_pred):.1f} bpm")
        if gt is not None:
            valid = np.isfinite(win_pred) & np.isfinite(win_ref)
            if valid.any():
                err = win_pred[valid] - win_ref[valid]
                mae = float(np.mean(np.abs(err)))
                rmse = float(np.sqrt(np.mean(err ** 2)))
                print(f"  MAE:          {mae:.2f} bpm")
                print(f"  RMSE:         {rmse:.2f} bpm")

    if args.plot or args.save_plot:
        times_s = np.arange(n) / fps
        if args.save_plot:
            dc_ac_path = args.save_plot.with_name(args.save_plot.stem + "_dc_ac.png")
        else:
            dc_ac_path = None
        make_dc_ac_plot(times_s, G, fps, save_path=dc_ac_path)
        make_plots(times_s, pulse, args.method.upper(), fps, filtered, f, pxx, global_hr,
                   win_t, win_pred, win_ref, gt, save_path=args.save_plot)


if __name__ == "__main__":
    main()
