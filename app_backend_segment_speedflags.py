import sys
import numpy as np
import cv2
import math
import subprocess
import os
import shutil
import tempfile
from pathlib import Path
#from matplotlib import pyplot as plt

THRESHOLD_RATIO = 2000
MIN_AVG_RED = 45 #(recommend 40:blue water)
MAX_HUE_SHIFT = 100 #(recommend 200:blue water)
BLUE_MAGIC_VALUE = 1.2 #(recommend 0.9:blue water)
SAMPLE_SECONDS = 2 # Extracts color correction from every N seconds
SAMPLE_WINDOW_SAMPLES = 15  # Number of sampled frames to mean per timepoint (must be odd)
#alpha = 0.75 # Contrast control (1.0-3.0)
#beta = 0 # Brightness control (0-100)
clip_hist_percent_in=0.3
shadow_amount_percent= 0.7 #Increase shadows 0 ~ 1 (recommend 0.6:blue water)
shadow_tone_percent=1
shadow_radius=0
highlight_amount_percent=0 ##decrease high lights 0 ~ 1 (recommend 0.2:blue water)
highlight_tone_percent=0
highlight_radius=0
USE_FAST_HS = True
FAST_HS_MAP_SCALE = 0.25

def mux_audio_from_source(source_video_path: str, corrected_video_path: str, final_output_path: str):
    ffmpeg = find_ffmpeg()

    # Map video from corrected, audio from source (audio optional)
    cmd = [
        ffmpeg, "-y",
        "-i", corrected_video_path,
        "-i", source_video_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        final_output_path
    ]

    # Capture stderr to avoid spam; raise on error
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed (exit {p.returncode}):\n{p.stderr}")

def resource_path(relative_name: str) -> str:
    base = getattr(sys, "_MEIPASS", str(Path(__file__).resolve().parent))
    return str(Path(base) / relative_name)

def find_ffmpeg() -> str:
    # 1) Bundled ffmpeg inside PyInstaller (preferred)
    bundled = resource_path("ffmpeg")
    if os.name == "nt":
        bundled = bundled + ".exe"

    if Path(bundled).exists():
        return bundled

    # 2) System ffmpeg on PATH
    ff = shutil.which("ffmpeg")
    if ff:
        return ff

    raise FileNotFoundError("ffmpeg not found (bundle it or install it on PATH).")

def ffmpeg_trim_segment(source_video_path: str, start_sec: float, duration_sec: float, trimmed_path: str) -> None:
    """
    Trim a segment from source_video_path into trimmed_path.

    Uses ffmpeg. Attempts stream-copy first (fast) and falls back to re-encode for
    better keyframe accuracy/compatibility if stream-copy fails.
    """
    ffmpeg = find_ffmpeg()

    start_sec = max(0.0, float(start_sec))
    duration_sec = max(0.0, float(duration_sec))
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")

    # Fast path: stream copy (may be off by a GOP / keyframe)
    cmd_copy = [
        ffmpeg, "-y",
        "-hwaccel", "auto",
        "-ss", f"{start_sec}",
        "-i", source_video_path,
        "-t", f"{duration_sec}",
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c", "copy",
        "-avoid_negative_ts", "1",
        trimmed_path,
    ]
    p = subprocess.run(cmd_copy, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode == 0 and Path(trimmed_path).exists() and Path(trimmed_path).stat().st_size > 0:
        return

    # Fallback: re-encode for reliable trimming
    cmd_reencode = [
        ffmpeg, "-y",
        "-ss", f"{start_sec}",
        "-i", source_video_path,
        "-t", f"{duration_sec}",
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        trimmed_path,
    ]
    p2 = subprocess.run(cmd_reencode, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p2.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed (exit {p2.returncode}):\n{p2.stderr}")


def hs_correction_fast(img_rgb, map_scale=FAST_HS_MAP_SCALE):
    """
    Approximate shadows/highlights:
    - compute Y (luma) at lower resolution
    - build shadow/highlight maps there
    - upscale corrected Y back to full resolution
    """
    shadow_amount_percent = globals().get("shadow_amount_percent", 0.7)
    shadow_tone_percent   = globals().get("shadow_tone_percent", 1.0)
    highlight_amount_percent = globals().get("highlight_amount_percent", 0.0)
    highlight_tone_percent   = globals().get("highlight_tone_percent", 0.0)

    shadow_tone = max(1.0, shadow_tone_percent * 255.0)
    highlight_tone = 255.0 - highlight_tone_percent * 255.0

    shadow_gain = 1.0 - float(shadow_amount_percent)
    highlight_gain = 1.0 + float(highlight_amount_percent) * 6.0

    # LUTs
    t = np.arange(256, dtype=np.float32)
    LUT_shadow = (1.0 - np.power(1.0 - t / 255.0, shadow_gain)) * 255.0
    LUT_shadow = np.clip(LUT_shadow + 0.5, 0, 255).astype(np.uint8)

    LUT_highlight = np.power(t / 255.0, highlight_gain) * 255.0
    LUT_highlight = np.clip(LUT_highlight + 0.5, 0, 255).astype(np.uint8)

    # Work in YCrCb (cheap)
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Y = ycrcb[..., 0]

    # Downscale Y for map computation
    if map_scale != 1.0:
        Ys = cv2.resize(Y, (0, 0), fx=map_scale, fy=map_scale, interpolation=cv2.INTER_AREA)
    else:
        Ys = Y

    Ys_f = Ys.astype(np.float32)

    # Shadow map
    shadow_map = 255.0 - Ys_f * 255.0 / shadow_tone
    shadow_map[Ys_f >= shadow_tone] = 0.0
    shadow_map = np.clip(shadow_map / 255.0, 0.0, 1.0)

    # Highlight map
    denom = max(1.0, (255.0 - highlight_tone))
    highlight_map = 255.0 - (255.0 - Ys_f) * 255.0 / denom
    highlight_map[Ys_f <= highlight_tone] = 0.0
    highlight_map = np.clip(highlight_map / 255.0, 0.0, 1.0)

    # Apply LUTs
    Ys_u8 = Ys
    iH = (1.0 - shadow_map) * Ys_f + shadow_map * LUT_shadow[Ys_u8].astype(np.float32)
    iH_u8 = np.clip(iH, 0, 255).astype(np.uint8)
    iH2 = (1.0 - highlight_map) * iH + highlight_map * LUT_highlight[iH_u8].astype(np.float32)
    Ys_corr = np.clip(iH2, 0, 255).astype(np.uint8)

    # Upscale corrected Y (already computed from low-res)
    if map_scale != 1.0:
        Y_corr = cv2.resize(Ys_corr, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Build a matching low-frequency baseline of the ORIGINAL Y
        Y_base_small = cv2.resize(Y, (0, 0), fx=map_scale, fy=map_scale, interpolation=cv2.INTER_AREA)
        Y_base = cv2.resize(Y_base_small, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        Y_corr = Ys_corr
        Y_base = Y

    # Detail-preserving: apply only low-frequency delta
    Y_f = Y.astype(np.float32)
    delta = (Y_corr.astype(np.float32) - Y_base.astype(np.float32))

    strength = 1.0  # you can expose this later if needed
    Y_new = np.clip(Y_f + strength * delta, 0, 255).astype(np.uint8)

    ycrcb[..., 0] = Y_new
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def hs_correction(
        img,
        shadow_amount_percent=None, shadow_tone_percent=None, shadow_radius=None,
        highlight_amount_percent=None, highlight_tone_percent=None, highlight_radius=None
):
    # Use current globals if not explicitly provided
    if shadow_amount_percent is None:
        shadow_amount_percent = globals().get("shadow_amount_percent", 0.7)
    if shadow_tone_percent is None:
        shadow_tone_percent = globals().get("shadow_tone_percent", 1.0)
    if shadow_radius is None:
        shadow_radius = globals().get("shadow_radius", 0)

    if highlight_amount_percent is None:
        highlight_amount_percent = globals().get("highlight_amount_percent", 0.0)
    if highlight_tone_percent is None:
        highlight_tone_percent = globals().get("highlight_tone_percent", 0.0)
    if highlight_radius is None:
        highlight_radius = globals().get("highlight_radius", 0)

    shadow_tone = shadow_tone_percent * 255
    highlight_tone = 255 - highlight_tone_percent * 255

    shadow_gain = 1 - shadow_amount_percent
    highlight_gain = 1 + highlight_amount_percent * 6

    height, width = img.shape[:2]
    img = img.astype(float)
    img_R, img_G, img_B = img[..., 2].reshape(-1), img[..., 1].reshape(-1), img[..., 0].reshape(-1)

    img_Y = .3 * img_R + .59 * img_G + .11 * img_B
    img_U = -img_R * .168736 - img_G * .331264 + img_B * .5
    img_V = img_R * .5 - img_G * .418688 - img_B * .081312

    #shadow_map = 255 - img_Y * 255 / shadow_tone
    #shadow_map[np.where(img_Y >= shadow_tone)] = 0
    if shadow_tone <= 0:
        shadow_map = np.zeros_like(img_Y)
    else:
        shadow_map = 255 - img_Y * 255 / shadow_tone
        shadow_map[img_Y >= shadow_tone] = 0
    

    highlight_map = 255 - (255 - img_Y) * 255 / (255 - highlight_tone)
    highlight_map[np.where(img_Y <= highlight_tone)] = 0

    t = np.arange(256)
    LUT_shadow = (1 - np.power(1 - t * (1 / 255), shadow_gain)) * 255
    LUT_shadow = np.maximum(0, np.minimum(255, np.int_(LUT_shadow + .5)))
    LUT_highlight = np.power(t * (1 / 255), highlight_gain) * 255
    LUT_highlight = np.maximum(0, np.minimum(255, np.int_(LUT_highlight + .5)))

    shadow_map = shadow_map * (1 / 255)
    highlight_map = highlight_map * (1 / 255)

    iH = (1 - shadow_map) * img_Y + shadow_map * LUT_shadow[np.int_(img_Y)]
    iH = (1 - highlight_map) * iH + highlight_map * LUT_highlight[np.int_(iH)]
    img_Y = iH

    output_R = np.int_(img_Y + 1.402 * img_V + .5)
    output_G = np.int_(img_Y - .34414 * img_U - .71414 * img_V + .5)
    output_B = np.int_(img_Y + 1.772 * img_U + .5)

    #output = np.row_stack([output_B, output_G, output_R]).T.reshape(height, width, 3)
    output = np.vstack((output_B, output_G, output_R)).T.reshape(height, width, 3)
    output = np.minimum(np.maximum(output, 0), 255).astype(np.uint8)
    return output


def automatic_brightness_and_contrast(image, clip_hist_percent=None):
    if clip_hist_percent is None:
        clip_hist_percent = globals().get("clip_hist_percent_in", 0.3)

    # Clamp to sane range (0..100)
    clip_hist_percent = max(0.0, min(float(clip_hist_percent), 100.0))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # hist_size = len(hist)

    # accumulator = [hist[0].item()]
    # for index in range(1, hist_size):
    #     accumulator.append(accumulator[index - 1] + hist[index].item())

    # maximum = accumulator[-1]
    # if maximum <= 0:
    #     # Empty/invalid image histogram; return identity transform
    #     return (image, 1.0, 0.0)

    # # Convert percent to absolute count; clip from both ends
    # clip = (clip_hist_percent / 100.0) * maximum / 2.0

    # # Find left cut
    # minimum_gray = 0
    # while minimum_gray < hist_size - 1 and accumulator[minimum_gray] < clip:
    #     minimum_gray += 1

    # # Find right cut (bounded so we never go negative)
    # maximum_gray = hist_size - 1
    # while maximum_gray > 0 and accumulator[maximum_gray] >= (maximum - clip):
    #     maximum_gray -= 1

    # # If bounds collapse, do nothing (avoid divide-by-zero / nonsense)
    # if maximum_gray <= minimum_gray:
    #     return (image, 1.0, 0.0)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    acc = np.cumsum(hist)
    maximum = acc[-1]
    if maximum <= 0:
        return (image, 1.0, 0.0)

    clip = (clip_hist_percent / 100.0) * maximum / 2.0

    minimum_gray = int(np.searchsorted(acc, clip))
    maximum_gray = int(np.searchsorted(acc, maximum - clip) - 1)

    if maximum_gray <= minimum_gray:
        return (image, 1.0, 0.0)

    alpha = 255.0 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


def convert_video_to_audio_ffmpeg(input_video_path, output_ext="mp3"):
    print("Extracting audio...")
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(input_video_path)
    subprocess.call(["ffmpeg", "-y", "-i", input_video_path, f"{filename}.{output_ext}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)

def merge_audio_video(output_video_path, input_video_path, output_ext="mp3"):
    filename, ext = os.path.splitext(input_video_path)
    filename1, ext = os.path.splitext(output_video_path)
    subprocess.call(["ffmpeg", "-i", output_video_path, "-i", f"{filename}.{output_ext}", "-c:v", 'copy', "-c:a", 'aac', f"{filename1}_audio.MP4"])
    os.remove(f"{filename}.{output_ext}")
    
def hue_shift_red(mat, h):

    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)

    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = 0.8*((0.587 - 0.587 * U + 0.330 * W)) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]

    return np.dstack([r, g, b])

def _emit_progress(stage: str, pct: float):
    # stage: "ANALYZE" or "PROCESS"
    pct = 0.0 if pct is None else float(pct)
    if pct < 0: pct = 0.0
    if pct > 100: pct = 100.0
    print(f"PROGRESS {stage} {pct:.2f}", flush=True)

def normalizing_interval(array):

    high = 255
    low = 0
    max_dist = 0

    for i in range(1, len(array)):
        dist = array[i] - array[i-1]
        if(dist > max_dist):
            max_dist = dist
            high = array[i]
            low = array[i-1]

    return (low, high)

def apply_filter(mat, filt):

    r = mat[..., 0]
    g = mat[..., 1]
    b = mat[..., 2]

    r = r * filt[0] + g*filt[1] + b*filt[2] + filt[4]*255
    g = g * filt[6] + filt[9] * 255
    b = b * filt[12] + filt[14] * 255

    filtered_mat = np.dstack([r, g, b])
    filtered_mat = np.clip(filtered_mat, 0, 255).astype(np.uint8)

    return filtered_mat

def get_filter_matrix(mat):

    mat = cv2.resize(mat, (256, 256))

    # Get average values of RGB
    avg_mat = np.array(cv2.mean(mat)[:3], dtype=np.uint8)
    
    # Find hue shift so that average red reaches MIN_AVG_RED
    new_avg_r = avg_mat[0]
    hue_shift = 0
    while(new_avg_r < MIN_AVG_RED):

        shifted = hue_shift_red(avg_mat, hue_shift)
        new_avg_r = np.sum(shifted)
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            new_avg_r = MIN_AVG_RED

    # Apply hue shift to whole image and replace red channel
    shifted_mat = hue_shift_red(mat, hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    new_r_channel = np.clip(new_r_channel, 0, 255)
    mat[..., 0] = new_r_channel

    # Get histogram of all channels
    hist_r = hist = cv2.calcHist([mat], [0], None, [256], [0,256])
    hist_g = hist = cv2.calcHist([mat], [1], None, [256], [0,256])
    hist_b = hist = cv2.calcHist([mat], [2], None, [256], [0,256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat.shape[0]*mat.shape[1])/THRESHOLD_RATIO
    for x in range(256):
        
        if hist_r[x] < threshold_level:
            normalize_mat[x][0] = x

        if hist_g[x] < threshold_level:
            normalize_mat[x][1] = x

        if hist_b[x] < threshold_level:
            normalize_mat[x][2] = x

    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])


    shifted = hue_shift_red(np.array([1, 1, 1]), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    red_gain = 256 / (adjust_r_high - adjust_r_low)
    green_gain = 256 / (adjust_g_high - adjust_g_low)
    blue_gain = 256 / (adjust_b_high - adjust_b_low)

    redOffset = (-adjust_r_low / 256) * red_gain
    greenOffset = (-adjust_g_low / 256) * green_gain
    blueOffset = (-adjust_b_low / 256) * blue_gain

    adjust_red = shifted_r * red_gain
    adjust_red_green = shifted_g * red_gain
    adjust_red_blue = shifted_b * red_gain * BLUE_MAGIC_VALUE

    return np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain, 0, 0, greenOffset,
        0, 0, blue_gain, 0, blueOffset,
        0, 0, 0, 1, 0,
    ])

def correct(mat):
    original_mat = mat.copy()

    filter_matrix = get_filter_matrix(mat)
    
    corrected_mat = apply_filter(original_mat, filter_matrix)
    corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)

    return corrected_mat

def correct_frame_full(
    frame_bgr: np.ndarray,
    *,
    filter_rgb_override: np.ndarray | None = None,
    disable_auto_contrast: bool = False,
    clip_hist_percent: float | None = None,
    use_fast_hs: bool | None = None,
    fast_hs_map_scale: float | None = None,
) -> np.ndarray:
    """Apply the same per-frame pipeline as video processing to a single BGR frame.

    This is intended for GUI frame-scrub preview.

    Pipeline (mirrors process_video):
      1) Compute filter matrix from the frame (get_filter_matrix)
      2) apply_filter (expects RGB ordering)
      3) optional auto brightness/contrast, but only apply it to the green channel
         (preserve red + blue channels)
      4) shadows/highlights correction (fast or full)
    """

    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty frame")

    # ---- 1) Filter matrix + apply_filter (RGB ordering) ----
    rgb_view = frame_bgr[..., ::-1]  # BGR -> RGB view
    filt_src = filter_rgb_override if (filter_rgb_override is not None and getattr(filter_rgb_override, 'size', 0) > 0) else rgb_view
    filt = get_filter_matrix(filt_src)
    corrected_rgb = apply_filter(rgb_view, filt)
    corrected_bgr = corrected_rgb[..., ::-1]  # RGB -> BGR
    corrected_bgr = np.ascontiguousarray(corrected_bgr)

    # ---- 2) Auto-contrast (optional) ----
    if clip_hist_percent is None:
        clip_hist_percent = globals().get("clip_hist_percent_in", 0.3)

    if not bool(disable_auto_contrast):
        auto_tmp, alpha, beta = automatic_brightness_and_contrast(
            corrected_bgr, clip_hist_percent=float(clip_hist_percent)
        )
        # Use raw alpha/beta (no temporal smoothing for single-frame preview)
        auto_result = cv2.convertScaleAbs(corrected_bgr, alpha=float(alpha), beta=float(beta))

        # Preserve original blue+red, replace green with auto-contrast adjusted green
        red_maintained = corrected_bgr.copy()
        red_maintained[..., 1] = auto_result[..., 1]
    else:
        red_maintained = corrected_bgr

    # ---- 3) Shadows/highlights ----
    if use_fast_hs is None:
        use_fast_hs = bool(globals().get("USE_FAST_HS", False))
    if fast_hs_map_scale is None:
        fast_hs_map_scale = float(globals().get("FAST_HS_MAP_SCALE", 0.5))

    red_maintained_rgb = red_maintained[..., ::-1]
    if bool(use_fast_hs):
        hscorrected_rgb = hs_correction_fast(red_maintained_rgb, map_scale=float(fast_hs_map_scale))
    else:
        hscorrected_rgb = hs_correction(red_maintained_rgb)

    out_bgr = hscorrected_rgb[..., ::-1]
    return np.ascontiguousarray(out_bgr)

def correct_image(input_path, output_path):
    mat = cv2.imread(input_path)
    rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    
    corrected_mat = correct(rgb_mat)

    cv2.imwrite(output_path, corrected_mat)
    
    preview = mat.copy()
    width = preview.shape[1] // 2
    preview[::, width:] = corrected_mat[::, width:]

    preview = cv2.resize(preview, (960, 540))

    return cv2.imencode('.png', preview)[1].tobytes()

def _sanitize_window_samples(n: int) -> int:
    """Ensure window sample count is a positive odd integer >= 1."""
    try:
        n = int(n)
    except Exception:
        return 1
    if n < 1:
        n = 1
    # force odd
    if n % 2 == 0:
        n += 1
    return n

def analyze_video(input_video_path, output_video_path, *, downsample: int = 1, fps_downsample: int = 1, max_fps: float = 0.0):
    
    # Initialize new video writer
    cap = cv2.VideoCapture(input_video_path)
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps_in <= 0.0:
        fps_in = 30.0
    # Use a rounded FPS for sampling cadence.
    fps = max(1, int(round(fps_in)))

    # FPS downsampling (drop frames + lower output FPS to preserve duration)
    fps_ds = int(fps_downsample or 1)
    if fps_ds < 1:
        fps_ds = 1

    # Optional max-FPS cap (converted to an additional FPS downsample factor)
    try:
        max_fps_val = float(max_fps or 0.0)
    except Exception:
        max_fps_val = 0.0
    if max_fps_val > 0.0 and fps_in > 0.0:
        # Ensure output FPS <= max_fps_val
        need = int(math.ceil(fps_in / max_fps_val))
        if need < 1:
            need = 1
        fps_ds = max(fps_ds, need)

    fps_out = fps_in / float(fps_ds)
    frame_count = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute stable downsample dimensions once (and keep them even where possible).
    ds = int(downsample or 1)
    if ds < 1:
        ds = 1
    base_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    base_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    ds_w = max(1, base_w // ds) if base_w else 0
    ds_h = max(1, base_h // ds) if base_h else 0

    # yuv420p prefers even dimensions; enforce even sizes so analysis matches processing.
    if ds_w and (ds_w % 2 == 1):
        ds_w = max(1, ds_w - 1)
    if ds_h and (ds_h % 2 == 1):
        ds_h = max(1, ds_h - 1)
    
    # Get filter matrices for every 10th frame
    filter_matrix_indexes = []
    filter_matrices = []
    sampled_rgb_frames = []  # RGB frames at sample points (downsampled if enabled)
    count = 0
    
    print("Analyzing...")
    last_pct = -1
    while(cap.isOpened()):
        
        count += 1  
        #print(f"{count} frames", end="\r")
        ret, frame = cap.read()
        # Optional downsample for faster analysis (matches processing). Use stable dims.
        if ret and ds > 1 and ds_w and ds_h:
            frame = cv2.resize(frame, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        if not ret:
            # End video read if we have gone beyond reported frame count
            if count >= frame_count:
                break

            # Failsafe to prevent an infinite loop
            if count >= 1e6:
                break
            # If a frame read fails unexpectedly, skip and try to continue.
            continue

        # Pick sample frames every N seconds (defer filter computation until we can apply a windowed mean)
        if count % int(round(fps * SAMPLE_SECONDS)) == 0:
            mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            filter_matrix_indexes.append(count)
            sampled_rgb_frames.append(mat)
            if frame_count > 0:
                pct = (100.0 * count) / frame_count
                ipct = int(pct)
                if ipct != last_pct:
                    last_pct = ipct
                    _emit_progress("ANALYZE", pct)

        yield count
        
    cap.release()
    # Compute filter matrices using a mean over a window of sampled frames.
    # This avoids expensive seeking/decoding while still stabilizing the estimate.
    filter_matrices = []
    n_samples = len(sampled_rgb_frames)
    if n_samples > 0:
        win = int(SAMPLE_WINDOW_SAMPLES) if int(SAMPLE_WINDOW_SAMPLES) > 0 else 1
        if win % 2 == 0:
            win += 1  # enforce odd
        half = win // 2
        for j in range(n_samples):
            lo = max(0, j - half)
            hi = min(n_samples, j + half + 1)
            # Mean in float32 then clip back to uint8 for filter estimation
            mean_rgb = np.mean(np.stack(sampled_rgb_frames[lo:hi], axis=0).astype(np.float32), axis=0)
            mean_rgb_u8 = np.clip(mean_rgb, 0, 255).astype(np.uint8)
            filter_matrices.append(get_filter_matrix(mean_rgb_u8))
    else:
        # No samples collected; keep empty arrays (caller should handle)
        filter_matrices = []

    # Build a interpolation function to get filter matrix at any given frame
    filter_matrices = np.array(filter_matrices)

    yield {
        "input_video_path": input_video_path,
        "output_video_path": output_video_path,
        "fps": float(fps_out),
        "fps_in": float(fps_in),
        "fps_downsample": int(fps_ds),
        "frame_count": count,
        "frame_count_out": int(math.ceil(count / float(fps_ds))) if fps_ds > 1 else int(count),
        "filters": filter_matrices,
        "filter_indices": filter_matrix_indexes,
        "downsample": int(downsample) if downsample else 1,
        "downsample_w": int(ds_w) if ds_w else None,
        "downsample_h": int(ds_h) if ds_h else None,
    }
def process_video_segment(
    input_path: str,
    start_sec: float,
    duration_sec: float,
    params: dict,
    output_path: str,
    progress_cb=None
):
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)

    """
    Process only a segment [start_sec, start_sec + duration_sec]
    """    

def start_ffmpeg_video_encoder(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "veryfast",
    pix_fmt_in: str = "bgr24",
    pix_fmt_out: str = "yuv420p",
):
    ffmpeg = find_ffmpeg()

    # Raw video frames will be written to stdin.
    # If the declared geometry does not match the byte stream, FFmpeg can
    # produce a "mosaic/tiled" image (multiple small frames packed into one).
    # Using explicit -video_size/-framerate helps avoid misinterpretation.
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", pix_fmt_in,
        "-video_size", f"{width}x{height}",
        "-framerate", str(float(fps)),
        "-i", "pipe:0",
        "-an",  # no audio in this pass; you already mux audio later :contentReference[oaicite:2]{index=2}
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(int(crf)),
        "-pix_fmt", pix_fmt_out,
        "-movflags", "+faststart",
        output_path,
    ]

    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return p

def process_video(
    video_data,
    yield_preview=False,
    *,
    auto_contrast_every_n_frames=None,
    precompute_filters=True,
    # Filter-matrix temporal stabilisation
    filter_smooth_alpha: float = 0.0,
    filter_max_delta: float = 0.0,
    # Auto-contrast temporal stabilisation (applied to alpha/beta)
    ac_smooth_alpha: float = 0.0,
    ac_max_delta_alpha: float = 0.0,
    ac_max_delta_beta: float = 0.0,
    disable_auto_contrast: bool = False,
):
    """
    Process a video using the analyzed filter matrices in `video_data`.

    Speed optimizations (drop-in, backward-compatible):
      - Cache auto-contrast (alpha/beta) and recompute only every N frames (default: ~once per second).
      - Optionally precompute interpolated filter matrices for all frames (bounded by a safety threshold).

    Args:
        video_data: dict from analyze_video()
        yield_preview: if True, yields (percent, preview_png_bytes)
        auto_contrast_every_n_frames: int or None. If None, defaults to max(1, int(fps)).
        precompute_filters: bool. If True, precompute per-frame filter coefficients when safe.
    """
    cap = cv2.VideoCapture(video_data["input_video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_data['input_video_path']}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ds = int(video_data.get("downsample", 1) or 1)
    if ds < 1:
        ds = 1

    # Downsampled output dimensions must be stable and must match the encoder's
    # declared frame size. Also ensure even dimensions for yuv420p.
    out_w = max(1, frame_width // ds)
    out_h = max(1, frame_height // ds)
    if out_w % 2 == 1:
        out_w = max(2, out_w - 1)
    if out_h % 2 == 1:
        out_h = max(2, out_h - 1)
    # Optional: force a specific output geometry (e.g., exact --max-height) while maintaining aspect ratio.
    try:
        fw = int(video_data.get("force_out_w", 0) or 0)
        fh = int(video_data.get("force_out_h", 0) or 0)
    except Exception:
        fw = fh = 0
    if fw > 0 and fh > 0:
        out_w = max(2, int(fw))
        out_h = max(2, int(fh))
        if out_w % 2 == 1:
            out_w = max(2, out_w - 1)
        if out_h % 2 == 1:
            out_h = max(2, out_h - 1)

    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # new_video = cv2.VideoWriter(video_data["output_video_path"], fourcc, video_data["fps"], (frame_width, frame_height))
    # if not new_video.isOpened():
    #     cap.release()
    #     raise RuntimeError(f"Failed to open VideoWriter for: {video_data['output_video_path']}")

    # Use FFmpeg for encoding instead of OpenCV VideoWriter (much faster, better threading)
    enc = start_ffmpeg_video_encoder(
        output_path=video_data["output_video_path"],
        width=out_w,
        height=out_h,
        fps=video_data["fps"],
        codec="libx264",
        crf=18,
        preset="veryfast",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
    )
    if enc.stdin is None:
        cap.release()
        raise RuntimeError("Failed to open ffmpeg stdin pipe for encoding.")


    filter_matrices = np.asarray(video_data["filters"], dtype=np.float32)
    filter_indices = np.asarray(video_data["filter_indices"], dtype=np.float32)

    filter_matrix_size = int(filter_matrices.shape[1])

    frame_count = int(video_data.get("frame_count", 0)) or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps_ds = int(video_data.get("fps_downsample", 1) or 1)
    if fps_ds < 1:
        fps_ds = 1
    frame_count_out = int(video_data.get("frame_count_out", 0) or 0)
    if frame_count_out <= 0:
        frame_count_out = int(math.ceil(frame_count / float(fps_ds))) if (frame_count > 0 and fps_ds > 1) else int(frame_count)
    fps = float(video_data.get("fps", 0) or cap.get(cv2.CAP_PROP_FPS) or 0)
    if auto_contrast_every_n_frames is None:
        auto_contrast_every_n_frames = max(1, int(fps) if fps > 0 else 30)

    # Safety bound to avoid huge memory use on very long videos:
    # 200k frames * 20 coeffs * 4 bytes ~ 16 MB
    PRECOMPUTE_MAX_FRAMES = 200_000

    filters_per_frame = None
    if precompute_filters and frame_count > 0 and frame_count <= PRECOMPUTE_MAX_FRAMES and len(filter_indices) >= 2:
        frames = np.arange(1, frame_count + 1, dtype=np.float32)
        filters_per_frame = np.empty((frame_count, filter_matrix_size), dtype=np.float32)
        for j in range(filter_matrix_size):
            filters_per_frame[:, j] = np.interp(frames, filter_indices, filter_matrices[:, j]).astype(np.float32)

    def get_interpolated_filter_matrix(frame_number: int):
        if filters_per_frame is not None and 1 <= frame_number <= filters_per_frame.shape[0]:
            return filters_per_frame[frame_number - 1]
        # Fallback: interpolate on-demand (slower)
        return np.array([np.interp(frame_number, filter_indices, filter_matrices[:, j]) for j in range(filter_matrix_size)], dtype=np.float32)

    print("Processing...")
    last_pct = -1

    # Cache for auto-contrast coefficients (applied via cv2.convertScaleAbs)
    alpha = 1.0
    beta = 0.0
    ac_initialized = False

    # Previous filter for temporal stabilisation (EMA + clamp)
    prev_filt = None

    src_count = 0
    out_count = 0
    while cap.isOpened():
        # Read/skip frames for FPS downsampling efficiently.
        ret, frame_bgr = cap.read()
        if not ret:
            break
        src_count += 1

        if fps_ds > 1:
            # Keep only every Nth frame (0-based).
            if ((src_count - 1) % fps_ds) != 0:
                # We already decoded this frame; continue. Subsequent frames will be read in-loop.
                continue

        out_count += 1
        # src_count is the frame number in the original stream; use it for filter interpolation.
        src_frame_number = src_count

        # Optional spatial downsample for speed. IMPORTANT: resize to the fixed encoder size.
        if ds > 1:
            frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)


        # Progress
        if frame_count_out > 0:
            pct = (100.0 * out_count) / frame_count_out
            ipct = int(pct)
            if ipct != last_pct:
                last_pct = ipct
                _emit_progress("PROCESS", pct)

        # # Apply the per-frame filter (operate in RGB because apply_filter() expects RGB ordering in this script)
        # rgb_mat = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # filt = get_interpolated_filter_matrix(src_frame_number)
        # corrected_rgb = apply_filter(rgb_mat, filt)
        # corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)

        # Apply the per-frame filter without cvtColor:
        # apply_filter() expects RGB ordering, so create an RGB view of the BGR frame via channel swap.
        rgb_view = frame_bgr[..., ::-1]  # BGR -> RGB view (no color conversion math)
        filt_raw = np.asarray(get_interpolated_filter_matrix(src_frame_number), dtype=np.float32)

        # Option 2: EMA smoothing (temporal inertia) + Option 3: per-frame delta clamp
        # Both operate in filter-coefficient space (float32). Values:
        #   filter_smooth_alpha: 0 disables, typical 0.01..0.20
        #   filter_max_delta:    0 disables, typical 0.005..0.05 (depends on footage)
        # filt_raw = np.asarray(filt_raw, dtype=np.float32)

        if prev_filt is None:
            filt_use = filt_raw
        else:
            filt_use = filt_raw

            smooth_a = float(filter_smooth_alpha or 0.0)
            if smooth_a > 0.0:
                # EMA: move a fraction towards the new estimate
                filt_use = prev_filt + smooth_a * (filt_use - prev_filt)

            max_d = float(filter_max_delta or 0.0)
            if max_d > 0.0:
                # Clamp per-coefficient delta. Note offsets are multiplied by 255 in apply_filter(),
                # so they must be clamped much tighter to avoid visible per-frame jumps.
                max_vec = np.full_like(filt_use, max_d, dtype=np.float32)
                for k in (4, 9, 14):
                    if k < max_vec.size:
                        max_vec[k] = max_d / 255.0
                delta = filt_use - prev_filt
                delta = np.clip(delta, -max_vec, max_vec)
                filt_use = prev_filt + delta

        prev_filt = filt_use
        corrected_rgb = apply_filter(rgb_view, filt_use)          # returns RGB array
        corrected_bgr = corrected_rgb[..., ::-1]              # RGB -> BGR view

        # Ensure contiguous for downstream OpenCV ops (cvtColor removed, but OpenCV prefers contiguous arrays)
        corrected_bgr = np.ascontiguousarray(corrected_bgr)

        # Auto brightness/contrast: recompute only every N frames
        if disable_auto_contrast:
            auto_result = corrected_bgr
            # alpha = 1.0
            # beta = 0.0
        else:
            should_recompute = (out_count == 1) or (auto_contrast_every_n_frames > 0 and (out_count % auto_contrast_every_n_frames) == 0)
            if should_recompute:
                auto_tmp, alpha_new, beta_new = automatic_brightness_and_contrast(corrected_bgr)

                # Apply EMA + clamp to alpha/beta if requested
                if not ac_initialized:
                    alpha = float(alpha_new)
                    beta = float(beta_new)
                    ac_initialized = True
                else:
                    # EMA
                    ac_a = float(ac_smooth_alpha or 0.0)
                    if ac_a > 0.0:
                        alpha = alpha + ac_a * (float(alpha_new) - alpha)
                        beta = beta + ac_a * (float(beta_new) - beta)

                    # Clamp
                    mda = float(ac_max_delta_alpha or 0.0)
                    mdb = float(ac_max_delta_beta or 0.0)
                    if mda > 0.0:
                        da = float(alpha_new) - alpha
                        da = max(-mda, min(mda, da))
                        alpha = alpha + da
                    if mdb > 0.0:
                        db = float(beta_new) - beta
                        db = max(-mdb, min(mdb, db))
                        beta = beta + db

                # Use the stabilised alpha/beta (not the raw auto_tmp)
                auto_result = cv2.convertScaleAbs(corrected_bgr, alpha=alpha, beta=beta)
            else:
                auto_result = cv2.convertScaleAbs(corrected_bgr, alpha=alpha, beta=beta)

        # Preserve original blue+red channels, replace green with auto-contrast adjusted green (as per original logic)
        # (This avoids cv2.split/merge overhead.)
        red_maintained = corrected_bgr.copy()
        red_maintained[..., 1] = auto_result[..., 1]  # green channel

        red_maintained_rgb = red_maintained[..., ::-1]  # BGR → RGB view

        if USE_FAST_HS:
            hscorrected_rgb = hs_correction_fast(red_maintained_rgb, map_scale=FAST_HS_MAP_SCALE)
        else:
            hscorrected_rgb = hs_correction(red_maintained_rgb)


        corrected_mat_final = hscorrected_rgb[..., ::-1]  # RGB → BGR
        corrected_mat_final = np.ascontiguousarray(corrected_mat_final)

        # Shadow/Highlight correction expects RGB in this script
        # red_maintained_rgb = cv2.cvtColor(red_maintained, cv2.COLOR_BGR2RGB)
        # hscorrected_rgb = hs_correction(red_maintained_rgb)
        # corrected_mat_final = cv2.cvtColor(hscorrected_rgb, cv2.COLOR_RGB2BGR)
        
        # Shadow/Highlight correction: hs_correction() indexes channels as BGR (R=img[...,2]),
        # so pass BGR directly and avoid two cvtColor calls.
        # corrected_mat_final = hs_correction(red_maintained)

        # Defensive: guarantee the frame written to FFmpeg exactly matches the
        # geometry declared in start_ffmpeg_video_encoder(). Any mismatch can
        # yield a "tiled/mosaic" output.
        if corrected_mat_final.shape[0] != out_h or corrected_mat_final.shape[1] != out_w:
            corrected_mat_final = cv2.resize(corrected_mat_final, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # new_video.write(corrected_mat_final)
        # Write raw BGR frame bytes to ffmpeg stdin
        if not corrected_mat_final.flags["C_CONTIGUOUS"]:
            corrected_mat_final = np.ascontiguousarray(corrected_mat_final)
        enc.stdin.write(corrected_mat_final.tobytes())

        if yield_preview:
            # Keep preview consistent with previous behavior: left original, right corrected (pre-HS) at half-res
            preview = frame_bgr.copy()
            width = preview.shape[1] // 2
            height = preview.shape[0] // 2
            preview[:, width:] = corrected_bgr[:, width:]
            preview = cv2.resize(preview, (width, height))
            percent = (100.0 * out_count / frame_count_out) if frame_count_out > 0 else 0.0
            yield percent, cv2.imencode('.png', preview)[1].tobytes()
        else:
            yield None

    cap.release()
    # Finalize ffmpeg encoding
    try:
        if enc.stdin:
            enc.stdin.close()
        stderr = enc.stderr.read().decode("utf-8", errors="replace") if enc.stderr else ""
        rc = enc.wait()
    finally:
        if enc.stderr:
            enc.stderr.close()

    if rc != 0:
        raise RuntimeError(f"ffmpeg encoder failed (exit {rc}):\n{stderr}")

def _downsample_from_max_res(base_w: int, base_h: int, *, max_w: int = 0, max_h: int = 0, max_dim: int = 0) -> int:
    """Compute an integer downsample factor so that (base_w//ds, base_h//ds) fits within the requested max resolution.

    - If max_dim is provided (>0), it is applied to the longer side (both max_w and max_h set to max_dim).
    - Returns ds >= 1.
    """
    try:
        bw = int(base_w or 0)
        bh = int(base_h or 0)
    except Exception:
        return 1
    if bw <= 0 or bh <= 0:
        return 1

    try:
        md = int(max_dim or 0)
    except Exception:
        md = 0
    if md > 0:
        max_w = md
        max_h = md

    try:
        mw = int(max_w or 0)
        mh = int(max_h or 0)
    except Exception:
        mw = 0
        mh = 0

    if mw <= 0 and mh <= 0:
        return 1
    if mw <= 0:
        mw = bw
    if mh <= 0:
        mh = bh

    mw = max(1, mw)
    mh = max(1, mh)

    # Need ds so that bw/ds <= mw and bh/ds <= mh  => ds >= bw/mw and ds >= bh/mh
    ds = int(math.ceil(max(bw / float(mw), bh / float(mh), 1.0)))
    return max(1, ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Blue correction for images/videos")

    sub = parser.add_subparsers(dest="mode", required=True)

    p_img = sub.add_parser("image", help="Process image")
    p_img.add_argument("source_image_path")
    p_img.add_argument("output_image_path")

    p_vid = sub.add_parser("video", help="Process video")
    p_vid.add_argument("source_video_path")
    p_vid.add_argument("output_video_path")
    p_vid.add_argument("--start-sec", type=float, default=None, help="Start time (seconds) for segment processing")
    p_vid.add_argument("--duration-sec", type=float, default=None, help="Duration (seconds) for segment processing")
    p_vid.add_argument("--downsample", type=int, default=1, help="Downsample factor for video processing (1=full, 2=half, 4=quarter).")
    p_vid.add_argument("--max-width", type=int, default=0, help="Max output width. If set, backend computes an additional downsample so output width <= max-width.")
    p_vid.add_argument("--max-height", type=int, default=0, help="Max output height. If set, backend computes an additional downsample so output height <= max-height.")
    p_vid.add_argument("--force-height", action="store_true", help="Force output height exactly to --max-height (aspect ratio preserved).")
    p_vid.add_argument("--max-dim", type=int, default=0, help="Max output dimension for the longer side. Convenience for setting both --max-width/--max-height.")
    p_vid.add_argument("--fps-downsample", type=int, default=1, help="FPS downsample factor (1=keep all frames, 2=every 2nd frame, etc.). Output FPS is divided by this factor to preserve duration.")
    p_vid.add_argument("--max-fps", type=float, default=0.0, help="Optional cap for output FPS. If >0, backend will increase the FPS downsample factor so that output FPS <= max-fps (while preserving duration).")

    # Shared tuning knobs
    def add_tuning(p):
        p.add_argument("--threshold-ratio", type=float, default=THRESHOLD_RATIO)
        p.add_argument("--min-avg-red", type=float, default=MIN_AVG_RED)
        p.add_argument("--max-hue-shift", type=float, default=MAX_HUE_SHIFT)
        p.add_argument("--blue-magic-value", type=float, default=BLUE_MAGIC_VALUE)
        p.add_argument("--sample-seconds", type=float, default=SAMPLE_SECONDS)
        p.add_argument("--sample-window-samples", type=int, default=SAMPLE_WINDOW_SAMPLES,
            help="Number of sampled frames to mean per timepoint (odd; e.g., 15 = 7 before, current, 7 after).")

        p.add_argument("--clip-hist-percent-in", type=float, default=clip_hist_percent_in)

        p.add_argument("--shadow-amount-percent", type=float, default=shadow_amount_percent)
        p.add_argument("--shadow-tone-percent", type=float, default=shadow_tone_percent)
        p.add_argument("--shadow-radius", type=int, default=shadow_radius)

        p.add_argument("--highlight-amount-percent", type=float, default=highlight_amount_percent)
        p.add_argument("--highlight-tone-percent", type=float, default=highlight_tone_percent)
        p.add_argument("--highlight-radius", type=int, default=highlight_radius)
        p.add_argument(
            "--fast-hs",
            action="store_true",
            help="Use fast shadow/highlight correction (downsampled maps, full-res output)")
        p.add_argument(
    "--fast-hs-map-scale",
            type=float,
            default=0.1,
            help="Map scale for --fast-hs (0.1..1.0). Smaller is faster; 1.0 equals full-res maps.")
# Performance knobs (optional)
# 0 means "auto" (defaults to ~once per second based on FPS)
        p.add_argument("--auto-contrast-every-n-frames", type=int, default=0,
            help="Recompute auto-contrast histogram every N frames (0=auto, ~fps). Larger is faster, less adaptive.")

        # Auto-contrast temporal stabilisation (applies EMA/clamp to alpha/beta)
        p.add_argument("--ac-smooth-alpha", type=float, default=0.0,
            help="EMA smoothing factor for auto-contrast alpha/beta (0 disables; typical 0.05..0.30).")
        p.add_argument("--ac-max-delta-alpha", type=float, default=0.0,
            help="Clamp per-update change in auto-contrast alpha (0 disables; typical 0.01..0.10).")
        p.add_argument("--ac-max-delta-beta", type=float, default=0.0,
            help="Clamp per-update change in auto-contrast beta (0 disables; typical 0.5..5).")
        p.add_argument("--disable-auto-contrast", action="store_true",
            help="Disable auto brightness/contrast stage.")

        # Filter-matrix temporal stabilisation
        p.add_argument("--filter-smooth-alpha", type=float, default=0.0,
            help="EMA smoothing factor for per-frame filter matrices (0 disables; typical 0.01..0.20).")
        p.add_argument("--filter-max-delta", type=float, default=0.0,
            help="Clamp the per-frame change applied to filter-matrix coefficients (0 disables; typical 0.005..0.05).")
# By default we precompute per-frame filter coefficients when safe.
# Use --no-precompute-filters to reduce memory at the cost of speed.
        p.add_argument("--no-precompute-filters", action="store_true",
            help="Disable precomputing per-frame filter coefficients (lower memory, slower).")

    add_tuning(p_img)
    add_tuning(p_vid)

    args = parser.parse_args()
    USE_FAST_HS = bool(getattr(args, "fast_hs", False))
    FAST_HS_MAP_SCALE = float(getattr(args, "fast_hs_map_scale", 0.5) or 0.5)
    # Clamp to sane range
    if FAST_HS_MAP_SCALE < 0.1:
        FAST_HS_MAP_SCALE = 0.1
    if FAST_HS_MAP_SCALE > 1.0:
        FAST_HS_MAP_SCALE = 1.0
    # Apply args to globals used throughout the pipeline
    THRESHOLD_RATIO = args.threshold_ratio
    MIN_AVG_RED = args.min_avg_red
    MAX_HUE_SHIFT = args.max_hue_shift
    BLUE_MAGIC_VALUE = args.blue_magic_value
    SAMPLE_SECONDS = args.sample_seconds

    SAMPLE_WINDOW_SAMPLES = args.sample_window_samples
    clip_hist_percent_in = args.clip_hist_percent_in

    shadow_amount_percent = args.shadow_amount_percent
    shadow_tone_percent = args.shadow_tone_percent
    shadow_radius = args.shadow_radius

    highlight_amount_percent = args.highlight_amount_percent
    highlight_tone_percent = args.highlight_tone_percent
    highlight_radius = args.highlight_radius
# Performance
    auto_contrast_every_n_frames = int(getattr(args, "auto_contrast_every_n_frames", 0) or 0)
    if auto_contrast_every_n_frames <= 0:
        auto_contrast_every_n_frames = None  # auto inside process_video()
    disable_auto_contrast = bool(getattr(args, "disable_auto_contrast", False))
    precompute_filters = not bool(getattr(args, "no_precompute_filters", False))

    filter_smooth_alpha = float(getattr(args, "filter_smooth_alpha", 0.0) or 0.0)
    if filter_smooth_alpha < 0.0:
        filter_smooth_alpha = 0.0
    if filter_smooth_alpha > 1.0:
        filter_smooth_alpha = 1.0

    filter_max_delta = float(getattr(args, "filter_max_delta", 0.0) or 0.0)
    if filter_max_delta < 0.0:
        filter_max_delta = 0.0

    ac_smooth_alpha = float(getattr(args, "ac_smooth_alpha", 0.0) or 0.0)
    if ac_smooth_alpha < 0.0:
        ac_smooth_alpha = 0.0
    if ac_smooth_alpha > 1.0:
        ac_smooth_alpha = 1.0

    ac_max_delta_alpha = float(getattr(args, "ac_max_delta_alpha", 0.0) or 0.0)
    if ac_max_delta_alpha < 0.0:
        ac_max_delta_alpha = 0.0

    ac_max_delta_beta = float(getattr(args, "ac_max_delta_beta", 0.0) or 0.0)
    if ac_max_delta_beta < 0.0:
        ac_max_delta_beta = 0.0

    # Filter-matrix stabilisation
    filter_smooth_alpha = float(getattr(args, "filter_smooth_alpha", 0.0) or 0.0)
    filter_smooth_alpha = max(0.0, min(1.0, filter_smooth_alpha))
    filter_max_delta = float(getattr(args, "filter_max_delta", 0.0) or 0.0)
    filter_max_delta = max(0.0, filter_max_delta)

    # Auto-contrast stabilisation
    ac_smooth_alpha = float(getattr(args, "ac_smooth_alpha", 0.0) or 0.0)
    ac_smooth_alpha = max(0.0, min(1.0, ac_smooth_alpha))
    ac_max_delta_alpha = float(getattr(args, "ac_max_delta_alpha", 0.0) or 0.0)
    ac_max_delta_alpha = max(0.0, ac_max_delta_alpha)
    ac_max_delta_beta = float(getattr(args, "ac_max_delta_beta", 0.0) or 0.0)
    ac_max_delta_beta = max(0.0, ac_max_delta_beta)
    video_downsample = int(getattr(args, "downsample", 1) or 1)
    fps_downsample = int(getattr(args, "fps_downsample", 1) or 1)
    if video_downsample < 1:
        video_downsample = 1
    if fps_downsample < 1:
        fps_downsample = 1

    if args.mode == "image":
        mat = cv2.imread(args.source_image_path)
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        corrected_mat = correct(mat)
        cv2.imwrite(args.output_image_path, corrected_mat)

    else:
        import tempfile
        import shutil
        from pathlib import Path

        # Create one temp working directory for optional trimming + video-only output
        tmp_dir = tempfile.mkdtemp(prefix="cc_")

        try:
            # Optionally trim input to a segment first
            source_for_processing = args.source_video_path
            if args.start_sec is not None and args.duration_sec is not None:
                start_sec = float(args.start_sec)
                duration_sec = float(args.duration_sec)
                if duration_sec <= 0:
                    raise ValueError("--duration-sec must be > 0 when --start-sec is provided")

                trimmed_src = str(Path(tmp_dir) / "trimmed_source.mp4")
                ffmpeg_trim_segment(args.source_video_path, start_sec, duration_sec, trimmed_src)
                source_for_processing = trimmed_src
            # Optional: derive downsample from a max-resolution constraint (in addition to --downsample).
            try:
                mw = int(getattr(args, 'max_width', 0) or 0)
                mh = int(getattr(args, 'max_height', 0) or 0)
                md = int(getattr(args, 'max_dim', 0) or 0)
            except Exception:
                mw = mh = md = 0
            if (mw > 0) or (mh > 0) or (md > 0):
                try:
                    _cap_probe = cv2.VideoCapture(source_for_processing)
                    bw = int(_cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    bh = int(_cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    _cap_probe.release()
                except Exception:
                    bw = bh = 0
                ds_box = _downsample_from_max_res(bw, bh, max_w=mw, max_h=mh, max_dim=md)
                video_downsample = max(1, int(video_downsample), int(ds_box))

            # 1) Analyze -> build video_data
            video_data = None
            max_fps = float(getattr(args, "max_fps", 0.0) or 0.0)

            for item in analyze_video(source_for_processing, args.output_video_path, downsample=video_downsample, fps_downsample=fps_downsample, max_fps=max_fps):
                if isinstance(item, dict):
                    video_data = item
                    break
            if video_data is None:
                raise RuntimeError("analyze_video() did not return video_data")
            # If requested, force the final output height exactly to --max-height (aspect ratio preserved).
            # This is applied at encode time (process_video) via a final resize to the target geometry.
            try:
                _force_h = bool(getattr(args, "force_height", False))
            except Exception:
                _force_h = False
            if _force_h:
                try:
                    _mh_force = int(getattr(args, "max_height", 0) or 0)
                except Exception:
                    _mh_force = 0
                if _mh_force > 0:
                    try:
                        _cap_probe2 = cv2.VideoCapture(source_for_processing)
                        _bw2 = int(_cap_probe2.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                        _bh2 = int(_cap_probe2.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                        _cap_probe2.release()
                    except Exception:
                        _bw2 = _bh2 = 0
                    if _bw2 > 0 and _bh2 > 0:
                        # Derive width from forced height, preserving aspect ratio.
                        _out_h_force = int(_mh_force)
                        _out_w_force = int(round(_bw2 * (_out_h_force / float(_bh2))))
                        # Ensure even dimensions for yuv420p.
                        if _out_h_force % 2 == 1:
                            _out_h_force = max(2, _out_h_force - 1)
                        if _out_w_force % 2 == 1:
                            _out_w_force = max(2, _out_w_force - 1)
                        _out_h_force = max(2, _out_h_force)
                        _out_w_force = max(2, _out_w_force)
                        video_data["force_out_h"] = _out_h_force
                        video_data["force_out_w"] = _out_w_force

            # 2) Write corrected video-only output to a temporary file
            out_final = args.output_video_path
            suffix = Path(out_final).suffix or ".mp4"
            tmp_video = str(Path(tmp_dir) / f"video_only{suffix}")

            # Ensure downstream uses temp output
            video_data["output_video_path"] = tmp_video

            [x for x in process_video(
                video_data,
                yield_preview=False,
                auto_contrast_every_n_frames=auto_contrast_every_n_frames,
                precompute_filters=precompute_filters,
                filter_smooth_alpha=filter_smooth_alpha,
                filter_max_delta=filter_max_delta,
                ac_smooth_alpha=ac_smooth_alpha,
                ac_max_delta_alpha=ac_max_delta_alpha,
                ac_max_delta_beta=ac_max_delta_beta,
                disable_auto_contrast=disable_auto_contrast,
            )]

            # 3) Mux source audio into corrected video (final output)
            mux_audio_from_source(source_for_processing, tmp_video, out_final)

            print("Done (video + audio).")

        finally:
            # 4) Cleanup temp files
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass



