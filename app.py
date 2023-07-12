from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    video = request.files['video']
    filename = secure_filename(video.filename)
    input_path = os.path.join('uploads', filename)
    output_path = os.path.join('uploads', 'enhanced_' + filename)
    video.save(input_path)

    # Call the video enhancement function
    enhance_video(input_path, output_path)

    return f'Video enhanced! <a href="{output_path}" target="_blank">Download</a>'

if __name__ == '__main__':
    app.run()
