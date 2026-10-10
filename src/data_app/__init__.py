"""Local sales analysis web application."""

from pathlib import Path
from threading import RLock

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, Conflict, HTTPException

from .pipeline import analyze, predict


def create_app(config=None):
    app = Flask(__name__)
    app.config.from_mapping(MAX_CONTENT_LENGTH=5 * 1024 * 1024)
    if config:
        app.config.update(config)
    lock = RLock()
    app.extensions["analysis"] = None

    @app.get("/")
    def home():
        return render_template("index.html")

    def run_analysis(content, source):
        try:
            result = analyze(content)
        except ValueError as error:
            raise BadRequest(str(error)) from error
        result.report["source"] = source
        with lock:
            app.extensions["analysis"] = result
        return result.report

    @app.post("/api/demo")
    def demo():
        content = (Path(__file__).parent / "data" / "demo_sales.csv").read_bytes()
        return run_analysis(content, "synthetic demo data")

    @app.post("/api/analyze")
    def upload():
        file = request.files.get("file")
        if file is None or not file.filename.lower().endswith(".csv"):
            raise BadRequest("Attach a CSV file in the file field")
        return run_analysis(file.read(), "uploaded CSV")

    @app.get("/api/report")
    def report():
        with lock:
            result = app.extensions["analysis"]
        if result is None:
            raise Conflict("Analyze a dataset first")
        return result.report

    @app.post("/api/predict")
    def prediction():
        with lock:
            result = app.extensions["analysis"]
        if result is None:
            raise Conflict("Analyze a dataset first")
        try:
            return predict(result, request.get_json())
        except ValueError as error:
            raise BadRequest(str(error)) from error

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.errorhandler(HTTPException)
    def http_error(error):
        response = error.get_response()
        response.data = jsonify(error={"code": error.code, "message": error.description}).data
        response.content_type = "application/json"
        return response

    return app
