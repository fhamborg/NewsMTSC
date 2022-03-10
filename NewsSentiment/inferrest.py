from flask import Flask, request

from NewsSentiment import TargetSentimentClassifier

app = Flask(__name__)

tsc = TargetSentimentClassifier()


@app.route("/infer", methods=["POST"])
def index():
    text_left = request.form["left"]
    target_mention = request.form["target"]
    text_right = request.form["right"]
    return {
        "result": tsc.infer(
            text_left=text_left, target_mention=target_mention, text_right=text_right
        )
    }


def start_rest_server(port=13273):
    print("starting server...")
    app.run(host="0.0.0.0", port=port)
    print("done")


if __name__ == "__main__":
    start_rest_server()
