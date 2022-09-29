from webapp import create_app
import argparse

app = create_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--host', type=str, default='127.0.0.1')
    #parser.add_argument('--port', type=int, default=5002)
    #parser.add_argument('--debug', action='store_true')
    #args = parser.parse_args()

    #app.run(host=args.host, port=args.port, debug=args.debug)
    app.run()