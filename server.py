from threading import Thread
from sys import stderr

import flask


t = None
app = flask.Flask(__name__)
counter = 0


@app.route('/')
def main():
    return f'Hello {counter}'


def serve():
    global t
    if t is None:
        t = Thread(target=app.run, kwargs={'port': 8448})
    t.start()
    print('So I kinda started', flush=True, file=stderr)


if __name__ == '__main__':
    serve()
