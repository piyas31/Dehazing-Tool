import numpy as np
import cv2
from scipy import fftpack

# ---------- Guided filter (fast box filter based) ----------
def box_filter(img, r):
    """Fast box filter using OpenCV boxFilter"""
    return cv2.boxFilter(img, -1, (r*2+1, r*2+1), normalize=True)

def guided_filter(I, p, r, eps):
    """
    I: guidance image (gray, float32)
    p: input image to be filtered (float32)
    r: radius
    eps: regularization
    returns: filtered image
    """
    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    corr_I = box_filter(I * I, r)
    corr_Ip = box_filter(I * p, r)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * I + mean_b
    return q

# ---------- Dark Channel Prior dehazing ----------
def dark_channel(img, sz=15):
    """
    img: uint8 BGR image (0-255)
    sz: patch size (odd)
    """
    # min over RGB channels
    b, g, r = cv2.split(img)
    min_img = cv2.min(cv2.min(r, g), b)
    # erode / minimum filter (using cv2.erode)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(min_img, kernel)
    return dark

def estimate_atmospheric_light(img, dark):
    """Estimate atmospheric light A from brightest pixels in dark channel."""
    # Flatten
    h, w = dark.shape
    numpx = max(int(h * w * 0.001), 1)  # top 0.1% pixels
    dark_vec = dark.reshape(h * w)
    img_vec = img.reshape(h * w, 3)

    # indices of largest dark channel values
    indices = np.argsort(dark_vec)[-numpx:]
    atms = img_vec[indices]
    A = np.max(atms, axis=0)
    return A.astype(np.float32)

def estimate_transmission(img, A, omega=0.95, sz=15):
    """
    img: uint8 BGR
    A: atmospheric light (3,)
    omega: usually 0.85-0.99 (higher -> more haze removed but risk artifacts)
    """
    # normalize by A
    norm = np.empty_like(img, dtype=np.float32)
    for c in range(3):
        norm[:, :, c] = img[:, :, c].astype(np.float32) / (A[c] + 1e-8)
    # compute dark channel of normalized image
    min_img = np.min(norm, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark_norm = cv2.erode(min_img, kernel)
    t = 1.0 - omega * dark_norm
    t = np.clip(t, 0.0, 1.0)
    return t

def recover_radiance(img, t, A, t0=0.1):
    """Recover scene radiance J = (I - A) / t + A"""
    img_f = img.astype(np.float32)
    t_rep = np.maximum(t, t0)[:, :, np.newaxis]
    J = (img_f - A) / t_rep + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J

# ---------- Full DCP pipeline -----------
def dehaze_dcp(img_bgr, omega=0.95, sz=15, guided_radius=40, guided_eps=1e-3, t0=0.1):
    """
    img_bgr: uint8 BGR (0-255)
    omega: haze-removal strength (0.7-0.99). Larger -> stronger removal.
    sz: patch size for dark channel (typically 15)
    guided_radius: radius for guided filter refinement
    guided_eps: eps for guided filter (small value)
    t0: lower bound for transmission (0.05-0.3)
    """

    dark = dark_channel(img_bgr, sz=sz)

    A = estimate_atmospheric_light(img_bgr, dark)
    raw_t = estimate_transmission(img_bgr, A, omega=omega, sz=sz)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    raw_t_f = raw_t.astype(np.float32)
    refined_t = guided_filter(gray, raw_t_f, r=guided_radius, eps=guided_eps)
    refined_t = np.clip(refined_t, 0.0, 1.0)

    J = recover_radiance(img_bgr, refined_t, A, t0=t0)
    J = histogram_stretch(J)
    J = unsharp_mask(J, amount=0.8, radius=1.0)

    return J

def histogram_stretch(img_bgr):
    out = img_bgr.copy().astype(np.uint8)
    for c in range(3):
        p = out[:, :, c]
        lo = np.percentile(p, 2)
        hi = np.percentile(p, 98)
        if hi - lo < 1:
            continue
        p = (p.astype(np.float32) - lo) * 255.0 / (hi - lo)
        p = np.clip(p, 0, 255).astype(np.uint8)
        out[:, :, c] = p
    return out

def unsharp_mask(img_bgr, amount=1.0, radius=1.0):
    img = img_bgr.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)
    mask = img - blurred
    sharp = img + amount * mask
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp

def frequency_enhance(img_bgr, amount=1.0):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = fftpack.fft2(img)
    fshift = fftpack.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    y = np.arange(rows) - crow
    x = np.arange(cols) - ccol
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    D = D / (D.max()+1e-8)
    H = 1 + amount * (D)
    fshift_filtered = fshift * H
    f_ishift = fftpack.ifftshift(fshift_filtered)
    img_back = np.real(fftpack.ifft2(f_ishift))
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = img_back
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return out

def adaptive_smooth_sharpen(img_bgr, sigma=1.0, amount=1.0):
    img = img_bgr.astype(np.float32)
    d = int(max(5, sigma * 5))
    smoothed = cv2.bilateralFilter(img, d=d, sigmaColor=75, sigmaSpace=sigma)
    blurred = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = cv2.subtract(smoothed, blurred)
    sharp = cv2.addWeighted(smoothed, 1.0, mask, amount, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp
