import os
import sys
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
from data.dataset import mag_pha_stft, mag_pha_istft

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
        enhance_audio_chunk(dummy, model, CONFIG, DEVICE)
        print("Warmup pass completed.")
    except Exception as e:
        print(f"Warmup notice: {e}")

    MODEL = model


def load_audio(file_path):
    """Loads any audio file robustly using soundfile or ffmpeg fallback."""
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


def stft_audio(audio, h):
    B, C, T = audio.shape
    audio = audio.reshape(B * C, T)
    mag, pha, com = mag_pha_stft(audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    return mag, pha, com


def istft_audio(mag, pha, h, shape):
    batch_size, channels, length = shape
    audio = mag_pha_istft(mag, pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor, length=length)
    audio = audio.reshape(batch_size, channels, audio.shape[-1])
    return audio


def enhance_audio_chunk(audio, generator, h, device):
    """Enhance a single chunk of audio."""
    audio = audio.unsqueeze(0).to(device)
    batch_size, channels, length = audio.shape
    mag, pha, _ = stft_audio(audio, h)
    mag_g, pha_g, _ = generator(mag, pha)
    audio_g = istft_audio(mag_g, pha_g, h, (batch_size, channels, length))
    return audio_g.squeeze(0).detach().cpu()


def enhance_audio_full(audio, generator, h, device, chunk_sec=24.0, overlap_sec=3.0):
    """
    Enhances full audio using Hann-smoothed overlap-add crossfade.
    - Uses 24-second chunks with 3-second Hann crossfade for seamless temporal context.
    - Short audio (<= 35s, including preview mode) runs in a single continuous pass.
    - Avoids phase cancellation boundary dips and volume fluctuations between chunks.
    - Zero-padding free tail processing preserves natural dynamics through to the end.
    """
    sr = h.sampling_rate
    channels, total_samples = audio.shape
    chunk_len = int(chunk_sec * sr)
    overlap = int(overlap_sec * sr)
    hop = chunk_len - overlap

    if total_samples <= sr * 35 or total_samples <= chunk_len + overlap:
        # Short audio or preview mode: run in a single continuous pass
        return enhance_audio_chunk(audio, generator, h, device)

    # Hann crossfade window for smooth overlap-add without phase dips
    hann = torch.hann_window(2 * overlap)
    fade_in = hann[:overlap]
    fade_out = hann[overlap:]
    window = torch.ones(chunk_len)
    window[:overlap] = fade_in
    window[-overlap:] = fade_out

    output = torch.zeros_like(audio)
    weights = torch.zeros(total_samples)

    # Determine chunk start positions ensuring the last chunk aligns to total_samples without zero-padding
    starts = list(range(0, total_samples - chunk_len, hop))
    if len(starts) == 0 or starts[-1] != (total_samples - chunk_len):
        starts.append(total_samples - chunk_len)

    num_chunks = len(starts)
    for idx, start in enumerate(starts):
        end = start + chunk_len
        chunk = audio[:, start:end]

        enhanced_chunk = enhance_audio_chunk(chunk, generator, h, device)

        cur_window = window.clone()
        if idx == 0:
            cur_window[:overlap] = 1.0
        if idx == num_chunks - 1:
            cur_window[-overlap:] = 1.0

        output[:, start:end] += enhanced_chunk * cur_window
        weights[start:end] += cur_window

    weights = weights.clamp(min=1e-6)
    output = output / weights
    return output


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


def generate_waveform_plot(audio_orig, audio_rest, sr, out_path):
    """Renders high-fidelity dual waveform comparison (Before vs After)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 3.8), dpi=110, facecolor='#121316', sharex=True)

    t = np.linspace(0, audio_orig.shape[-1] / sr, audio_orig.shape[-1])
    # Downsample for rendering speed if very large
    max_pts = 100000
    if len(t) > max_pts:
        step = len(t) // max_pts
        t = t[::step]
        orig_plot = audio_orig[:, ::step].numpy()
        rest_plot = audio_rest[:, ::step].numpy()
    else:
        orig_plot = audio_orig.numpy()
        rest_plot = audio_rest.numpy()

    # Original Waveform
    ax1.set_facecolor('#0f1115')
    ax1.plot(t, orig_plot[0], color='#00DFF8', alpha=0.9, linewidth=0.6, label='Left Channel')
    if orig_plot.shape[0] > 1:
        ax1.plot(t, orig_plot[1], color='#5be9fb', alpha=0.65, linewidth=0.5, label='Right Channel')
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel('Original', color='#00DFF8', fontsize=10, fontweight='bold')
    ax1.grid(True, color='#2d3748', alpha=0.5, linestyle=':')
    ax1.tick_params(colors='#7b8794')
    for spine in ax1.spines.values():
        spine.set_color('#2d3748')

    # Restored Waveform
    ax2.set_facecolor('#0f1115')
    ax2.plot(t, rest_plot[0], color='#20c997', alpha=0.85, linewidth=0.6, label='Left Channel')
    if rest_plot.shape[0] > 1:
        ax2.plot(t, rest_plot[1], color='#38d9a9', alpha=0.65, linewidth=0.5, label='Right Channel')
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_ylabel('Restored', color='#20c997', fontsize=10, fontweight='bold')
    ax2.grid(True, color='#2d3748', alpha=0.5, linestyle=':')
    ax2.tick_params(colors='#7b8794')
    for spine in ax2.spines.values():
        spine.set_color('#2d3748')

    # Time ticks
    total_sec = audio_orig.shape[-1] / sr
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

        # 1. Load Audio
        audio, sr = load_audio(input_save_path)
        orig_sr = sr
        channels = audio.shape[0]

        # Resample to 44.1 kHz if necessary
        if sr != CONFIG.sampling_rate:
            import torchaudio
            audio = torchaudio.functional.resample(audio, sr, CONFIG.sampling_rate)
            sr = CONFIG.sampling_rate

        # Preview mode: slice first 30 seconds
        if preview_mode and audio.shape[-1] > sr * 30:
            audio = audio[:, :sr * 30]

        total_duration = audio.shape[-1] / sr

        # Estimate original cutoff frequency (Spek style)
        audio_mono = audio.mean(dim=0).numpy()
        orig_cutoff = estimate_cutoff_frequency(audio_mono, sr)

        # 2. Run MPASCNet Enhancement
        with MODEL_LOCK:
            with torch.no_grad():
                enhanced = enhance_audio_full(audio, MODEL, CONFIG, DEVICE)
            enhanced = enhanced.clamp(-1.0, 1.0)

        process_time = time.time() - start_time

        # 3. Export Audio Files
        orig_wav_path = os.path.join(OUTPUT_DIR, f"{session_id}_orig.wav")
        rest_wav_path = os.path.join(OUTPUT_DIR, f"{session_id}_restored.wav")

        sf.write(orig_wav_path, audio.T.numpy(), sr)
        sf.write(rest_wav_path, enhanced.T.numpy(), sr)

        # 4. Generate Spek Spectrograms
        spek_orig_png = os.path.join(OUTPUT_DIR, f"{session_id}_spek_orig.png")
        spek_rest_png = os.path.join(OUTPUT_DIR, f"{session_id}_spek_rest.png")
        waveform_png = os.path.join(OUTPUT_DIR, f"{session_id}_waveform.png")

        min_sec_str = f"{int(total_duration // 60):02d}:{int(total_duration % 60):02d}"
        meta_orig = f"Original Audio • {min_sec_str} • {channels}ch • {orig_sr} Hz • Cutoff: ~{orig_cutoff/1000.0:.1f} kHz"
        meta_rest = f"MPASCNet Restored • {min_sec_str} • {channels}ch • {sr} Hz • Restored Bandwidth: 22.05 kHz"

        generate_spek_spectrogram(audio.numpy(), sr, f"Original: {orig_filename}", meta_orig, spek_orig_png, is_restored=False)
        generate_spek_spectrogram(enhanced.numpy(), sr, f"Restored: {orig_filename}", meta_rest, spek_rest_png, is_restored=True)

        # 5. Generate Waveform Plot
        generate_waveform_plot(audio, enhanced, sr, waveform_png)

        return jsonify({
            "success": True,
            "session_id": session_id,
            "filename": orig_filename,
            "duration_sec": round(total_duration, 2),
            "duration_formatted": min_sec_str,
            "channels": channels,
            "sample_rate": sr,
            "orig_cutoff_khz": round(orig_cutoff / 1000.0, 1),
            "restored_bandwidth_khz": 22.05,
            "process_time_sec": round(process_time, 2),
            "orig_audio_url": f"/api/audio/{os.path.basename(orig_wav_path)}",
            "rest_audio_url": f"/api/audio/{os.path.basename(rest_wav_path)}",
            "spek_orig_url": f"/api/image/{os.path.basename(spek_orig_png)}",
            "spek_rest_url": f"/api/image/{os.path.basename(spek_rest_png)}",
            "waveform_url": f"/api/image/{os.path.basename(waveform_png)}",
            "download_url": f"/api/download/{os.path.basename(rest_wav_path)}"
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
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


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
