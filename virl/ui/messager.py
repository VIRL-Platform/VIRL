import requests


class Messager(object):
    def __init__(self, cfg):
        port = cfg.PORT
        host = cfg.HOST
        self.url = f'http://{host}:{port}'
        # server_run(host, port)
        print('Initialized Messager')

    def send_image(self, query, image_id, image_data):
        response = requests.post(
            f'{self.url}/{query}',
            json={'image_id': image_id, 'image_data': image_data}
        )

        if response.status_code != 200:
            print(response.status_code, 'Unsuccessful image results sent')

    def send_image_list(self, query, image_id, image_data):
        response = requests.post(
            f'{self.url}/{query}',
            json={'image_id': image_id, 'image_data': image_data}
        )

        if response.status_code != 200:
            print(response.status_code, 'Unsuccessful image results sent')

    def send_text(self, query, text_id, text_data):
        response = requests.post(
            f'{self.url}/{query}',
            json={'text_id': text_id, 'text': text_data},
            timeout=3
        )

        if response.status_code != 200:
            print(response.status_code, 'Unsuccessful text results sent')

    def clear(self, elem_ids):
        response = requests.post(
            f'{self.url}/clear',
            json={'elem_ids': elem_ids},
            timeout=3
        )

        if response.status_code != 200:
            print(response.status_code, 'Unsuccessful clearing')
