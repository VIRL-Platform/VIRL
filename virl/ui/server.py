import secrets
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit


app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = secrets.token_hex(16)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/send_image', methods=['POST'])
@cross_origin()
def send_image():
    image_id = request.json.get('image_id')
    image_data = request.json.get('image_data')

    with app.app_context():

        socketio.emit('image', {"id": image_id, "image": image_data})

    return jsonify({'status': 'Image sent successfully'})


@app.route('/send_image_list', methods=['POST'])
@cross_origin()
def send_image_list():
    image_id = request.json.get('image_id')
    image_data = request.json.get('image_data')

    with app.app_context():

        socketio.emit('image_list', {"id_list": image_id, "image_list": image_data})

    return jsonify({'status': 'Image list sent successfully'})


@app.route('/send_text', methods=['POST'])
@cross_origin()
def send_text():
    text_id = request.json.get('text_id')
    text = request.json.get('text')

    with app.app_context():

        socketio.emit('text', {"id": text_id, "text": text})

    return jsonify({'status': 'Text sent successfully'})


@app.route('/clear', methods=['POST'])
@cross_origin()
def clear():
    elem_ids = request.json.get('elem_ids')

    with app.app_context():
        socketio.emit('clear', {"elem_ids": elem_ids})

    return jsonify({'status': 'Clear Successfully'})


def run(host="127.0.0.1", port=5000):
    socketio.run(app, debug=True, host=host, port=port)


if __name__ == '__main__':
    run()
