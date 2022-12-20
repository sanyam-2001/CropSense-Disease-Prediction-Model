from flask import Flask, request
import os
import calendar
import time
import predict

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello", 200


@app.route('/', methods=['POST'])
def create_item():
    data = request.get_json()
    print(data)
    return data


@app.route('/upload', methods=['POST'])
def upload_image():
    image_file = request.files['image']
    if image_file == None:
        return 'No image file provided', 400
    fileName = str(calendar.timegm(time.gmtime())) + image_file.filename
    image_file.save(os.path.join('./files', fileName))
    solutions = predict.predictPool(fileName)
    print(solutions)

    return 'Image uploaded successfully', 201


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
