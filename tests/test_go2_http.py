"""Real loopback HTTP transport, using no robot or model backend."""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from topomap_deploy.server import Service, make_handler


def test_http_auth_readiness_reset_and_stop():
    service = Service(None, lambda detector, goal: object(), ["chair"])
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, "local-test-token"))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def request(path, payload=None, authorized=True):
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["Authorization"] = "Bearer local-test-token"
        req = Request(url + path, None if payload is None else json.dumps(payload).encode(), headers)
        with urlopen(req, timeout=3.) as response:
            return json.load(response)

    try:
        with pytest.raises(HTTPError) as error:
            request("/health", authorized=False)
        assert error.value.code == 401
        assert request("/health")["protocol"] == 1
        session = request("/reset", {"goal": "chair"})["session"]
        assert request("/stop", {"session": session})["stopped"] is True
        with pytest.raises(HTTPError) as error:
            request("/stop", {"session": session})
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3.)
