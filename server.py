from threading import Thread

import flask


t = None
app = flask.Flask(__name__)
emb_map = None



import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app.logger.setLevel(logging.ERROR)


@app.route('/')
def main():
    if emb_map is None:
        return 'Hello World!'
    else:
        return '\n'.join(f'{w}: {vec}' for w, vec in emb_map.items())


def serve():
    global t
    if t is None:
        t = Thread(target=app.run, kwargs={'port': 8448})
    t.start()


if __name__ == '__main__':
    serve()
