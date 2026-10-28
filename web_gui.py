import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import re
import urllib.parse
import time
import json
import uuid
import subprocess
import threading
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import soundfile as sf
import logging
# Suppress noisy font lookup and werkzeug startup warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import flask.cli
flask.cli.show_server_banner = lambda *args, **kwargs: None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Typography font stack (Windows Segoe UI, macOS SF, Linux DejaVu Sans)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'DejaVu Sans', 'Segoe UI', 'Noto Sans', 'Liberation Sans', 'Helvetica Neue', 'Arial', 'sans-serif'
]
matplotlib.rcParams['font.monospace'] = [
    'DejaVu Sans Mono', 'Consolas', 'SF Mono', 'Menlo', 'Liberation Mono', 'monospace'
]
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory, render_template

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import AttrDict
from models.wapper import MPASCNet
from utils import load_checkpoint
from inference import enhance_audio, stft_audio, istft_audio

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoint', 'best')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'web_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250MB upload limit

# Global Model Cache
MODEL = None
DEVICE = None
CONFIG = None
MODEL_LOCK = threading.Lock()

# Spek exact color palette
SPEK_COLORS = [
    (0.00, '#000000'),  # -120 dB
    (0.12, '#030225'),  # -105 dB
    (0.24, '#05187b'),  # -90 dB
    (0.38, '#0155b4'),  # -74 dB
    (0.52, '#009a73'),  # -57 dB
    (0.68, '#4fba1e'),  # -38 dB
    (0.80, '#f2d400'),  # -24 dB
    (0.92, '#f53e00'),  # -10 dB
    (1.00, '#ffffff'),  # 0 dB
]
SPEK_CMAP = LinearSegmentedColormap.from_list('spek', [(pt[0], pt[1]) for pt in SPEK_COLORS])


def init_model():
    global MODEL, DEVICE, CONFIG
    if MODEL is not None:
        return

    print("Initializing MPASCNet engine...")
    with open(CONFIG_PATH) as f:
        config_data = json.load(f)
    CONFIG = AttrDict(config_data)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {DEVICE}")

    model = MPASCNet(CONFIG).to(DEVICE)
    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"Checkpoint not found at {CHECKPOINT_PATH}. Auto-downloading from Hugging Face...")
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve("https://huggingface.co/Xinghour/MPASCNet/resolve/main/best", CHECKPOINT_PATH)
        print("Checkpoint downloaded successfully.")

    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    sd = load_checkpoint(CHECKPOINT_PATH, DEVICE)
    model.load_state_dict(sd['generator'] if 'generator' in sd else sd)
    print("Checkpoint loaded successfully.")

    model.eval()

    # Warmup pass
    try:
        dummy = torch.zeros(2, CONFIG.sampling_rate * 2)
        enhance_audio(dummy, model, CONFIG, DEVICE)
        print("Warmup pass completed.")
    except Exception as e:
        print(f"Warmup notice: {e}")

    MODEL = model


def load_audio(file_path):
    """Loads any audio file using torchaudio first (their standard pipeline), with soundfile / ffmpeg fallback."""
    import torchaudio
    try:
        audio, sr = torchaudio.load(file_path)
        return audio.float(), sr
    except Exception:
        pass

    try:
        data, sr = sf.read(file_path)
        if data.ndim == 1:
            audio = torch.from_numpy(data[None, :]).float()
        else:
            audio = torch.from_numpy(data.T).float()
        return audio, sr
    except Exception as e:
        # Fallback to ffmpeg pipe
        cmd = [
            'ffmpeg', '-y', '-i', file_path,
            '-f', 'f32le', '-acodec', 'pcm_f32le',
            '-ar', '44100', '-ac', '2', '-'
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_bytes, _ = proc.communicate()
        if len(raw_bytes) == 0:
            raise RuntimeError(f"Failed to decode audio file {file_path}: {e}")
        audio_np = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, 2).T
        return torch.from_numpy(audio_np).float(), 44100


def enhance_audio_adaptive(audio, generator, h, device):
    """
    Intelligently runs inference:
    1. If the audio fits safely in GPU VRAM (up to 3.5 minutes on Colab 15GB T4 with memory-efficient attention):
       Runs in a single continuous unchunked pass using 100% official vanilla inference code.
       Zero cuts, zero crossfades, zero destructive phase interference.
    2. If duration exceeds safe VRAM limit or on CUDA OOM (e.g. ultra-long podcasts or DJ sets):
       Runs high-fidelity margin-chunked inference with 90s segments and 8s receptive margins.
    """
    total_samples = audio.shape[-1]
    sr = h.sampling_rate
    total_sec = total_samples / sr

    # Determine safe single-pass duration based on GPU VRAM
    max_safe_sec = 180.0  # Conservative default (fits easily in 12GB)
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            if total_vram_gb >= 28:
                max_safe_sec = 500.0  # RTX 5090 (32GB)
            elif total_vram_gb >= 22:
                max_safe_sec = 360.0  # RTX 3090 / 4090 (24GB)
            elif total_vram_gb >= 14:
                max_safe_sec = 210.0  # Tesla T4 (Colab 15GB, ~3.5 min continuous)
            elif total_vram_gb >= 10:
                max_safe_sec = 160.0  # RTX 3060 / 4070 (12GB)
            else:
                max_safe_sec = 90.0   # 8GB GPUs (~1.5 min continuous)
        except Exception:
            pass

    if total_sec <= max_safe_sec:
        try:
            return enhance_audio(audio, generator, h, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
            if "out of memory" not in str(oom_err).lower() and not isinstance(oom_err, torch.cuda.OutOfMemoryError):
                raise
            torch.cuda.empty_cache()
            print("VRAM limit reached during single pass. Falling back to high-fidelity margin chunking...")

    # High-fidelity margin-based chunking for ultra-long audio
    return enhance_audio_margin_chunked(audio, generator, h, device, segment_sec=90.0, margin_sec=8.0, crossfade_sec=4.0)


def enhance_audio_margin_chunked(audio, generator, h, device, segment_sec=90.0, margin_sec=8.0, crossfade_sec=4.0):
    sr = h.sampling_rate
    channels, total_samples = audio.shape
    seg_len = int(segment_sec * sr)
    margin = int(margin_sec * sr)
    crossfade = int(crossfade_sec * sr)
    hop = seg_len - crossfade

    fade_in = 0.5 * (1.0 - torch.cos(torch.linspace(0, np.pi, crossfade)))
    fade_out = 0.5 * (1.0 + torch.cos(torch.linspace(0, np.pi, crossfade)))

    output = torch.zeros_like(audio)
    weights = torch.zeros(total_samples)

    for start in range(0, total_samples, hop):
        end = min(start + seg_len, total_samples)
        valid_len = end - start

        left_pad = min(start, margin)
        right_pad = min(total_samples - end, margin)

        chunk_start = start - left_pad
        chunk_end = end + right_pad
        chunk = audio[:, chunk_start:chunk_end]

        enhanced = enhance_audio(chunk, generator, h, device)
        valid_enhanced = enhanced[:, left_pad : left_pad + valid_len]

        window = torch.ones(valid_len)
        if start > 0:
            window[:crossfade] = fade_in[:min(crossfade, valid_len)]
        if end < total_samples:
            window[-crossfade:] = fade_out[-min(crossfade, valid_len):]

        output[:, start:end] += valid_enhanced * window
        weights[start:end] += window
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return output / weights.clamp(min=1e-6)


def estimate_cutoff_frequency(audio_mono, sr, threshold_db=-85):
    """Estimates the high-frequency cutoff (where MP3 compression discarded high frequencies)."""
    try:
        f, _, Sxx = signal.spectrogram(audio_mono, fs=sr, window='hann', nperseg=2048, noverlap=1024)
        avg_spec = np.mean(Sxx, axis=1)
        avg_db = 10 * np.log10(np.maximum(avg_spec, 1e-12))
        max_db = np.max(avg_db)
        norm_db = avg_db - max_db

        # Search downwards from Nyquist to find where energy first rises above threshold
        cutoff_idx = len(norm_db) - 1
        while cutoff_idx > 0 and norm_db[cutoff_idx] < threshold_db:
            cutoff_idx -= 1

        cutoff_freq = f[cutoff_idx]
        return min(float(cutoff_freq), float(sr / 2))
    except Exception:
        return float(sr / 2)


def spek_spectrum_rgb(level):
    """
    Port of Dan Bruton's spectrum algorithm as used in alexkay/spek (spek-palette.cc).
    Maps level [0.0, 1.0] (-120 dB to 0 dB) to exact Spek RGB colors.
    Reference: https://github.com/alexkay/spek
    """
    level = np.clip(level, 0.0, 1.0) * 0.6625
    r = np.zeros_like(level)
    g = np.zeros_like(level)
    b = np.zeros_like(level)

    # 0 <= level < 0.15
    m1 = (level >= 0) & (level < 0.15)
    r[m1] = (0.15 - level[m1]) / (0.15 + 0.075)
    g[m1] = 0.0
    b[m1] = 1.0

    # 0.15 <= level < 0.275
    m2 = (level >= 0.15) & (level < 0.275)
    r[m2] = 0.0
    g[m2] = (level[m2] - 0.15) / (0.275 - 0.15)
    b[m2] = 1.0

    # 0.275 <= level < 0.325
    m3 = (level >= 0.275) & (level < 0.325)
    r[m3] = 0.0
    g[m3] = 1.0
    b[m3] = (0.325 - level[m3]) / (0.325 - 0.275)

    # 0.325 <= level < 0.5
    m4 = (level >= 0.325) & (level < 0.5)
    r[m4] = (level[m4] - 0.325) / (0.5 - 0.325)
    g[m4] = 1.0
    b[m4] = 0.0

    # 0.5 <= level <= 0.6625
    m5 = (level >= 0.5) & (level <= 0.6625)
    r[m5] = 1.0
    g[m5] = (0.6625 - level[m5]) / (0.6625 - 0.5)
    b[m5] = 0.0

    # Intensity correction
    cf = np.ones_like(level)
    m_cf = (level >= 0.0) & (level < 0.1)
    cf[m_cf] = level[m_cf] / 0.1

    rr = np.clip(r * cf, 0.0, 1.0)
    gg = np.clip(g * cf, 0.0, 1.0)
    bb = np.clip(b * cf, 0.0, 1.0)

    return np.stack([rr, gg, bb], axis=-1)


def generate_spek_spectrogram(audio_np, sr, title, meta_info, out_path, is_restored=False):
    """
    Renders an authentic Spek style acoustic spectrogram (reference: https://github.com/alexkay/spek):
    - Pure pitch black canvas
    - Exact C++ Dan Bruton spectrum palette from alexkay/spek
    - Exact Spek FFT normalization: power = |X|^2 / N^2
    - Spek dB range: -120 dB (Black) to 0 dB (Bright Red)
    - 0, 5, 10, 15, 20, 22 kHz frequency rulers with subtle dotted grid
    - Exact right-hand Spek colorbar
    """
    mono = audio_np.mean(axis=0) if audio_np.ndim > 1 else audio_np
    N = 2048
    total_samples = len(mono)
    target_frames = 1400
    hop = max(1, (total_samples - N) // target_frames)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N) / (N - 1.0)))

    powers = []
    for start in range(0, total_samples - N, hop):
        chunk = mono[start : start + N] * w
        X = np.fft.rfft(chunk, n=N)
        p = (X.real**2 + X.imag**2) / (N * N)
        powers.append(p)

    powers = np.array(powers).T  # shape: (1025, frames)
    db = 10.0 * np.log10(np.maximum(powers, 1e-15))
    levels = (np.clip(db, -120.0, 0.0) + 120.0) / 120.0
    rgb = spek_spectrum_rgb(levels)

    # Colorbar strip
    cbar_levels = np.linspace(1.0, 0.0, 256).reshape(-1, 1)
    cbar_rgb = spek_spectrum_rgb(cbar_levels)

    fig = plt.figure(figsize=(11, 7.6), dpi=110, facecolor='#000000')
    ax = fig.add_axes([0.07, 0.09, 0.83, 0.82])
    cax = fig.add_axes([0.915, 0.09, 0.018, 0.82])

    total_dur = total_samples / sr
    max_khz = sr / 2000.0

    ax.imshow(rgb, origin='lower', aspect='auto', extent=[0, total_dur, 0, max_khz], interpolation='nearest')
    ax.set_facecolor('#000000')

    # Borders
    for spine in ax.spines.values():
        spine.set_color('#888888')
        spine.set_linewidth(1.0)

    # Frequency ruler
    yticks = [0, 5, 10, 15, 20]
    if int(round(max_khz)) not in yticks and max_khz > 20:
        yticks.append(int(round(max_khz)))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{int(k)} kHz' for k in yticks], color='#cccccc', fontsize=9.5)
    ax.grid(True, axis='y', color='#555555', alpha=0.6, linestyle=':', linewidth=0.9)

    # Time ruler
    step = 5 if total_dur <= 30 else (10 if total_dur <= 60 else (30 if total_dur <= 300 else 60))
    xticks = np.arange(0, total_dur + 0.1, step)
    if total_dur - xticks[-1] > step * 0.4:
        xticks = np.append(xticks, total_dur)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(x//60)}:{int(x%60):02d}' for x in xticks], color='#cccccc', fontsize=9.5)
    ax.tick_params(colors='#cccccc', which='both', length=4)

    # Header metadata (Spek style)
    header_color = '#4ade80' if is_restored else '#ffffff'
    fig.text(0.07, 0.965, title, color=header_color, fontsize=11, fontweight='bold')
    fig.text(0.07, 0.930, meta_info, color='#aaaaaa', fontsize=9)

    # Right-hand Colorbar
    cax.imshow(cbar_rgb, aspect='auto', extent=[0, 1, -120, 0])
    cax.set_yticks([0, -20, -40, -60, -80, -100, -120])
    cax.set_yticklabels(['0 dB', '-20 dB', '-40 dB', '-60 dB', '-80 dB', '-100 dB', '-120 dB'], color='#cccccc', fontsize=9)
    cax.yaxis.tick_right()
    cax.set_xticks([])
    cax.tick_params(colors='#cccccc', which='both', length=3)
    for spine in cax.spines.values():
        spine.set_color('#888888')
        spine.set_linewidth(1.0)

    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)


def generate_waveform_plot(audio_orig, audio_rest, sr_orig, sr_rest, out_path):
    """Renders high-fidelity dual waveform comparison (Before vs After)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 3.8), dpi=110, facecolor='#121316', sharex=True)

    t1 = np.linspace(0, audio_orig.shape[-1] / sr_orig, audio_orig.shape[-1])
    t2 = np.linspace(0, audio_rest.shape[-1] / sr_rest, audio_rest.shape[-1])

    max_pts = 100000
    if len(t1) > max_pts:
        step1 = len(t1) // max_pts
        t1 = t1[::step1]
        orig_plot = audio_orig[:, ::step1].numpy()
    else:
        orig_plot = audio_orig.numpy()

    if len(t2) > max_pts:
        step2 = len(t2) // max_pts
        t2 = t2[::step2]
        rest_plot = audio_rest[:, ::step2].numpy()
    else:
        rest_plot = audio_rest.numpy()

    # Original Waveform
    ax1.set_facecolor('#0f1115')
    ax1.plot(t1, orig_plot[0], color='#00DFF8', alpha=0.9, linewidth=0.6, label='Left Channel')
    if orig_plot.shape[0] > 1:
        ax1.plot(t1, orig_plot[1], color='#5be9fb', alpha=0.65, linewidth=0.5, label='Right Channel')
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel('Original', color='#00DFF8', fontsize=10, fontweight='bold')
    ax1.grid(True, color='#2d3748', alpha=0.5, linestyle=':')
    ax1.tick_params(colors='#7b8794')
    for spine in ax1.spines.values():
        spine.set_color('#2d3748')

    # Restored Waveform
    ax2.set_facecolor('#0f1115')
    ax2.plot(t2, rest_plot[0], color='#20c997', alpha=0.85, linewidth=0.6, label='Left Channel')
    if rest_plot.shape[0] > 1:
        ax2.plot(t2, rest_plot[1], color='#38d9a9', alpha=0.65, linewidth=0.5, label='Right Channel')
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_ylabel('Restored', color='#20c997', fontsize=10, fontweight='bold')
    ax2.grid(True, color='#2d3748', alpha=0.5, linestyle=':')
    ax2.tick_params(colors='#7b8794')
    for spine in ax2.spines.values():
        spine.set_color('#2d3748')

    # Time ticks
    total_sec = max(audio_orig.shape[-1] / sr_orig, audio_rest.shape[-1] / sr_rest)
    step_time = 5 if total_sec <= 30 else (10 if total_sec <= 60 else (30 if total_sec <= 300 else 60))
    xticks = np.arange(0, total_sec + 0.1, step_time)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f'{int(x//60):02d}:{int(x%60):02d}' for x in xticks], color='#9da7b3', fontsize=9)
    ax2.set_xlabel('Time (mm:ss)', color='#9da7b3', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)


# Routes
@app.route('/favicon.ico')
def favicon():
    return ('', 204)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    has_checkpoint = os.path.isfile(CHECKPOINT_PATH)
    ckpt_size = os.path.getsize(CHECKPOINT_PATH) if has_checkpoint else 0

    return jsonify({
        "status": "ready" if has_checkpoint else "missing_checkpoint",
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_exists": has_checkpoint,
        "checkpoint_size_mb": round(ckpt_size / (1024 * 1024), 2),
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "sampling_rate": CONFIG.sampling_rate if CONFIG else 44100
    })


@app.route('/api/process', methods=['POST'])
def api_process():
    init_model()

    file = request.files.get('file')
    local_path = request.form.get('local_path', '').strip()
    preview_mode = request.form.get('preview_mode', 'false').lower() == 'true'
    keep_44k = request.form.get('keep_44k', 'false').lower() == 'true'

    if not file and not local_path:
        return jsonify({"error": "No audio file or local path provided."}), 400

    session_id = str(uuid.uuid4())[:8]

    # Save or locate source audio
    if file:
        orig_filename = file.filename
        input_ext = os.path.splitext(orig_filename)[1].lower() or '.mp3'
        input_save_path = os.path.join(OUTPUT_DIR, f"{session_id}_input{input_ext}")
        file.save(input_save_path)
    else:
        if not os.path.isfile(local_path):
            return jsonify({"error": f"Local file not found: {local_path}"}), 404
        orig_filename = os.path.basename(local_path)
        input_save_path = local_path

    try:
        start_time = time.time()

        # 1. Load Audio (strictly using official pipeline with robust fallback)
        audio, sr = load_audio(input_save_path)
        orig_sr = sr
        channels = audio.shape[0]
        original_length = audio.shape[-1]

        # Preview mode: slice first 30 seconds
        if preview_mode and audio.shape[-1] > sr * 30:
            audio = audio[:, :sr * 30]
            original_length = audio.shape[-1]

        total_duration = audio.shape[-1] / sr

        # Estimate original cutoff frequency (Spek style)
        audio_mono = audio.mean(dim=0).numpy()
        orig_cutoff = estimate_cutoff_frequency(audio_mono, sr)

        # 2. Run MPASCNet Enhancement - Exact official inference.py pipeline
        import torchaudio
        if sr != CONFIG.sampling_rate:
            audio_for_model = torchaudio.functional.resample(audio, sr, CONFIG.sampling_rate)
        else:
            audio_for_model = audio

        with MODEL_LOCK:
            peak = audio_for_model.abs().max().clamp(min=1e-8)
            with torch.no_grad():
                enhanced = enhance_audio_adaptive(audio_for_model / peak, MODEL, CONFIG, DEVICE) * peak

        # Match official inference.py: if input sampling rate != model sampling rate,
        # resample back to input sampling rate and match original_length,
        # unless user explicitly asked to keep 44.1 kHz.
        if sr != CONFIG.sampling_rate and not keep_44k:
            enhanced = torchaudio.functional.resample(enhanced, CONFIG.sampling_rate, sr)
            if enhanced.shape[-1] >= original_length:
                enhanced = enhanced[..., :original_length]
            else:
                enhanced = F.pad(enhanced, (0, original_length - enhanced.shape[-1]))
            out_sr = sr
        else:
            out_sr = CONFIG.sampling_rate if sr != CONFIG.sampling_rate and keep_44k else sr

        enhanced = enhanced.clamp(-1.0, 1.0)
        process_time = time.time() - start_time

        # 3. Export Audio Files
        base_name = os.path.splitext(orig_filename)[0]
        safe_base = re.sub(r'[\\/*?:"<>|]', '_', base_name).strip() or "audio"
        download_filename = f"{safe_base}_restored.wav"

        orig_wav_path = os.path.join(OUTPUT_DIR, f"{session_id}_orig.wav")
        rest_wav_path = os.path.join(OUTPUT_DIR, f"{session_id}_{download_filename}")

        torchaudio.save(orig_wav_path, audio.clamp(-1.0, 1.0), sr)
        torchaudio.save(rest_wav_path, enhanced, out_sr)

        # 4. Generate Spek Spectrograms
        spek_orig_png = os.path.join(OUTPUT_DIR, f"{session_id}_spek_orig.png")
        spek_rest_png = os.path.join(OUTPUT_DIR, f"{session_id}_spek_rest.png")
        waveform_png = os.path.join(OUTPUT_DIR, f"{session_id}_waveform.png")

        min_sec_str = f"{int(total_duration // 60):02d}:{int(total_duration % 60):02d}"
        meta_orig = f"Original Audio • {min_sec_str} • {channels}ch • {orig_sr} Hz • Cutoff: ~{orig_cutoff/1000.0:.1f} kHz"
        meta_rest = f"MPASCNet Restored • {min_sec_str} • {channels}ch • {out_sr} Hz • Restored Bandwidth: {out_sr/2000.0:.2f} kHz"

        generate_spek_spectrogram(audio.numpy(), sr, f"Original: {orig_filename}", meta_orig, spek_orig_png, is_restored=False)
        generate_spek_spectrogram(enhanced.numpy(), out_sr, f"Restored: {orig_filename}", meta_rest, spek_rest_png, is_restored=True)

        # 5. Generate Waveform Plot
        generate_waveform_plot(audio, enhanced, sr, out_sr, waveform_png)

        return jsonify({
            "success": True,
            "session_id": session_id,
            "filename": orig_filename,
            "download_filename": download_filename,
            "duration_sec": round(total_duration, 2),
            "duration_formatted": min_sec_str,
            "channels": channels,
            "sample_rate": out_sr,
            "orig_cutoff_khz": round(orig_cutoff / 1000.0, 1),
            "restored_bandwidth_khz": round(out_sr / 2000.0, 2),
            "process_time_sec": round(process_time, 2),
            "orig_audio_url": f"/api/audio/{os.path.basename(orig_wav_path)}",
            "rest_audio_url": f"/api/audio/{os.path.basename(rest_wav_path)}",
            "spek_orig_url": f"/api/image/{os.path.basename(spek_orig_png)}",
            "spek_rest_url": f"/api/image/{os.path.basename(spek_rest_png)}",
            "waveform_url": f"/api/image/{os.path.basename(waveform_png)}",
            "download_url": f"/api/download/{os.path.basename(rest_wav_path)}?name={urllib.parse.quote(download_filename)}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype="audio/wav")


@app.route('/api/image/<filename>')
def serve_image(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype="image/png")


@app.route('/api/download/<filename>')
def download_audio(filename):
    custom_name = request.args.get('name')
    if not custom_name:
        custom_name = re.sub(r'^[a-f0-9]{8}_', '', filename)
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True, download_name=custom_name)


if __name__ == '__main__':
    import argparse
    import re
    parser = argparse.ArgumentParser(description="MPASCNet Web GUI")
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 7860)), help="Port to listen on")
    parser.add_argument('--colab-url', type=str, default=os.environ.get('COLAB_PROXY_URL', ''), help="Direct Google Colab Proxy URL")
    parser.add_argument('--share', action='store_true', help="Create public Cloudflare tunnel link")
    args = parser.parse_args()

    init_model()

    is_colab = os.path.exists('/content') or 'google.colab' in sys.modules
    target_url = (args.colab_url or os.environ.get('COLAB_PROXY_URL', '')).strip()

    if args.share and sys.platform.startswith('linux'):
        # Cloudflare Quick Tunnel (Zero config, zero password)
        if not os.path.isfile('/usr/local/bin/cloudflared'):
            try:
                subprocess.run(['curl', '-s', '-L', '-o', '/usr/local/bin/cloudflared', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['chmod', '+x', '/usr/local/bin/cloudflared'], check=True)
            except Exception:
                pass

        def run_cf_tunnel():
            try:
                p = subprocess.Popen(['cloudflared', 'tunnel', '--url', f'http://127.0.0.1:{args.port}'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in p.stdout:
                    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if match:
                        print(f"  🔗 Cloudflare Public URL: {match.group(0)}\n")
            except Exception:
                pass

        threading.Thread(target=run_cf_tunnel, daemon=True).start()

    print(f"\n=======================================================")
    if target_url:
        print(f"  MPASCNet Cloud Studio is running in Google Colab!")
        print(f"  👉 PUBLIC SHARE URL: {target_url}")
    elif is_colab:
        print(f"  MPASCNet Cloud Studio is running in Google Colab on port {args.port}!")
    else:
        print(f"  MPASCNet Web GUI is running!")
        print(f"  👉 Local URL: http://localhost:{args.port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=args.port, debug=False)
