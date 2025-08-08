from flask import Flask, render_template, request, send_from_directory
import os
from PIL import Image
import numpy as np
import uuid
from utils import (
    frequency_enhance,
    histogram_stretch,
    adaptive_smooth_sharpen,
    dehaze_dcp
)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 


@app.route('/', methods=['GET', 'POST'])
def index():
    original_image = None
    processed_image = None
    method = 'dcp'

    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template('index.html', original_image=None, processed_image=None, method=method)

        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        pil = Image.open(file.stream).convert('RGB')
        img = np.array(pil)[:, :, ::-1].copy()  

        method = request.form.get('method', 'dcp')

        omega = float(request.form.get('omega') or 0.95)
        patch = int(request.form.get('patch') or 15)
        guided_r = int(request.form.get('guided_r') or 40)
        guided_eps = float(request.form.get('guided_eps') or 1e-3)
        t0 = float(request.form.get('t0') or 0.1)

        apply_freq = request.form.get('freq') == 'on'
        apply_hist = request.form.get('hist') == 'on'
        apply_adapt = request.form.get('adapt') == 'on'
        freq_amt = float(request.form.get('freq_amt') or 1.0)
        adapt_sigma = float(request.form.get('adapt_sigma') or 1.0)
        adapt_sharp = float(request.form.get('adapt_sharp') or 1.0)

        if method == 'dcp':
            out = dehaze_dcp(img, omega=omega, sz=patch, guided_radius=guided_r, guided_eps=guided_eps, t0=t0)
        else:
            out = img.copy()
            if apply_freq:
                out = frequency_enhance(out, amount=freq_amt)
            if apply_hist:
                out = histogram_stretch(out)
            if apply_adapt:
                out = adaptive_smooth_sharpen(out, sigma=adapt_sigma, amount=adapt_sharp)

        out_rgb = out[:, :, ::-1]
        Image.fromarray(out_rgb).save(filepath, quality=95)
        orig_filename = f"{uuid.uuid4().hex}_orig.jpg"
        orig_filepath = os.path.join(app.config['UPLOAD_FOLDER'], orig_filename)
        pil.save(orig_filepath)

        original_image_url = f"/{orig_filepath}"
        processed_image_url = f"/{filepath}"

        return render_template(
            'index.html',
            original_image=original_image_url,
            processed_image=processed_image_url,
            method=method
        )

    return render_template('index.html', original_image=None, processed_image=None, method=method)




@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
