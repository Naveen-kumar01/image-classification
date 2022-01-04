from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from skimage import transform
from skimage.transform import resize
import tensorflow
from wsgiref import simple_server
from flask_cors import CORS, cross_origin

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Define a flask app
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_image(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

# Model saved with Keras model.save()
MODEL_PATH = 'custom_model.h5'


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        file = request.files['file']
        try:
            if file and allowed_file(file.filename):
                filename = file.filename
                if os.listdir("static") is not None:
                    for i in os.listdir('static'):
                        os.remove(os.path.join('static', i))
                file_path = os.path.join('static', filename)
                file.save(file_path)
                img = read_image(file_path)
                # Predict the class of an image
                model1 = load_model(MODEL_PATH)
                x = model1.predict(img)
                classes = ['Camel', 'Eagle', 'Fox', 'Frog', 'Giraffe', 'Kangaroo', 'Lion', 'Rhinoceros', 'Shark',
                           'Snake', 'Tiger', 'Zebra']
                product = classes[x.argmax()]
                score = str(x.max() * 100) + ' %'
                return render_template('predict.html', product=product, score=score, user_image=file_path)
        except:
            return "Unable to read the file. Please check if the file extension is correct."


port = int(os.getenv("PORT", 5000))
if __name__ == "__main__":
    host = '127.0.0.1'
    #port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
