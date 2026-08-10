"""Serving-layer regression tests.

Two failure modes are pinned down here, both of which made browser requests
hang for minutes against the chat page:

1. SSE responses set ``Connection: keep-alive``. That is a hop-by-hop header,
   forbidden for WSGI applications by PEP 3333. The Werkzeug development
   server tolerates it, but a compliant server (waitress) rejects the response
   with a 500, which silently turns a stream into an error.

2. The app was served by the Werkzeug development server, which speaks
   HTTP/1.0. Every response closes the connection, and a request written onto
   a connection that is being closed is dropped without a response or a reset.
"""

import importlib
import inspect
import unittest
from unittest import mock

from flask import Flask, Response

# 'from ghc_api import main' resolves to the main() function re-exported by the
# package, not the module, so import the module explicitly.
main_module = importlib.import_module("ghc_api.main")

# WSGI forbids these on an application response; waitress raises AssertionError.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}

SSE_MODULES = [
    "ghc_api/sse/base.py",
    "ghc_api/routes/openai.py",
    "ghc_api/routes/anthropic.py",
    "ghc_api/routes/agent.py",
]


class HopByHopHeaderTest(unittest.TestCase):
    def test_no_module_sets_a_hop_by_hop_header(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        for rel in SSE_MODULES:
            source = (root / rel).read_text()
            for header in HOP_BY_HOP:
                self.assertNotIn(
                    f'"{header.title()}": ',
                    source,
                    f"{rel} sets the hop-by-hop header {header.title()}; "
                    "PEP 3333 forbids it and waitress answers 500",
                )

    def test_streaming_response_is_accepted_by_a_pep3333_checker(self):
        """A response like the SSE handlers build must survive wsgiref's validator."""
        from wsgiref.validate import validator

        app = Flask(__name__)

        @app.route("/stream")
        def stream():
            return Response(
                (f"data: {i}\n\n" for i in range(3)),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        environ_headers = {}

        def start_response(status, headers, exc_info=None):
            environ_headers.update({k.lower(): v for k, v in headers})
            return lambda s: None

        from werkzeug.test import EnvironBuilder

        environ = EnvironBuilder(path="/stream").get_environ()
        environ.setdefault("wsgi.errors", __import__("io").StringIO())
        result = validator(app.wsgi_app)(environ, start_response)
        try:
            body = b"".join(result)
        finally:
            result.close()

        self.assertIn(b"data: 0", body)
        for header in HOP_BY_HOP:
            self.assertNotIn(header, environ_headers)


class ServeAppTest(unittest.TestCase):
    """serve_app must not use the development server for normal runs."""

    def setUp(self):
        self.app = mock.Mock()

    def test_uses_waitress_by_default(self):
        with mock.patch.dict("sys.modules", {"waitress": mock.Mock()}) as modules:
            serve = modules["waitress"].serve
            main_module.serve_app(self.app, host="127.0.0.1", port=1234, debug=False)

        serve.assert_called_once()
        kwargs = serve.call_args.kwargs
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 1234)
        self.assertGreaterEqual(kwargs["threads"], 4)
        self.app.run.assert_not_called()

    def test_channel_timeout_outlives_a_quiet_stream(self):
        with mock.patch.dict("sys.modules", {"waitress": mock.Mock()}) as modules, \
                mock.patch.object(main_module.state, "upstream_read_timeout", 1800):
            main_module.serve_app(self.app, host="127.0.0.1", port=1234, debug=False)
            kwargs = modules["waitress"].serve.call_args.kwargs

        self.assertGreaterEqual(kwargs["channel_timeout"], 1800)

    def test_debug_still_uses_the_development_server(self):
        main_module.serve_app(self.app, host="127.0.0.1", port=1234, debug=True)

        self.app.run.assert_called_once()
        self.assertTrue(self.app.run.call_args.kwargs["debug"])

    def test_falls_back_when_waitress_is_missing(self):
        real_import = __import__

        def no_waitress(name, *args, **kwargs):
            if name == "waitress":
                raise ImportError("no waitress")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=no_waitress):
            main_module.serve_app(self.app, host="127.0.0.1", port=1234, debug=False)

        self.app.run.assert_called_once()

    def test_main_serves_through_serve_app(self):
        """main() must not call app.run() directly any more."""
        source = inspect.getsource(main_module.main)
        self.assertIn("serve_app(", source)
        self.assertNotIn("app.run(", source)


if __name__ == "__main__":
    unittest.main()
