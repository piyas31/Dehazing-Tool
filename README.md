# Dehazing-Tool

A Python Flask web application that removes fog and haze from urban images using advanced image processing techniques like Dark Channel Prior dehazing, frequency domain enhancement, histogram stretching, and adaptive smoothing & sharpening. Ideal for improving visibility in traffic monitoring and construction inspection under poor weather conditions.

---

## Features

- Upload foggy or hazy urban images via web interface  
- Advanced dehazing using Dark Channel Prior algorithm  
- Additional image enhancement: frequency enhancement, histogram stretching, adaptive smoothing & sharpening  
- Download processed clear images  
- Easy-to-use, minimal UI built with Tailwind CSS inline styles  

---

## Requirements

- Python 3.12+  
- Flask  
- numpy  
- opencv-python  
- scipy  
- Pillow  
- matplotlib  

---

## Installation & Running Locally

1. Clone the repository: `git clone https://github.com/piyas31/dehazing-tool.git` and `cd dehazing-tool`

2. Create and activate a virtual environment:

- On Windows (PowerShell): `python -m venv venv` then `.\venv\Scripts\activate`

- On Linux/MacOS: `python3 -m venv venv` then `source venv/bin/activate`

3. Install dependencies: `pip install -r requirements.txt`

4. Run the app: `python app.py`

5. Open your browser and visit: `http://127.0.0.1:5000`

---

## Usage

- Upload a foggy or hazy urban image via the file input  
- Select the dehazing method (DCP or Pipeline) from the dropdown  
- Adjust dehazing parameters like omega (strength), patch size, guided filter radius, etc. as needed  
- Click the “Process” button to remove fog and enhance the image  
- View the processed clear image below the form  
- Download the enhanced image using the download button  
- Repeat the process to try different images or parameter settings

---

## License

© Zakiuzzman Piyas

---

Feel free to open issues or submit pull requests for improvements!

---

Thank you for using LightMender!
