"""The per-process internal token lets ghc-api's own pages and agent call the LLM endpoints when auth is on."""

import unittest
from unittest import mock

from ghc_api.app import create_app
from ghc_api.auth import INTERNAL_TOKEN, INTERNAL_USER_ID, get_user_registry, require_auth
from ghc_api.state import state


class InternalTokenTest(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(state, "enable_auth", True)
        patch.start()
        self.addCleanup(patch.stop)
        self.app = create_app()
        self.client = self.app.test_client()

    def test_require_auth_accepts_it_as_the_internal_user(self):
        for headers in ({"Authorization": f"Bearer {INTERNAL_TOKEN}"}, {"x-api-key": INTERNAL_TOKEN}):
            with self.subTest(headers=list(headers)), self.app.test_request_context(headers=headers):
                from flask import request

                self.assertEqual(require_auth(request).user_id, INTERNAL_USER_ID)

    def test_protected_routes_accept_it(self):
        # /proxy/* has its own auth gate; both gates go through require_auth.
        self.assertEqual(self.client.get("/proxy/models").status_code, 401)
        response = self.client.get("/proxy/models", headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"})
        self.assertEqual(response.status_code, 200)

    def test_a_user_cannot_register_the_internal_name(self):
        record, error = get_user_registry().create_pending(INTERNAL_USER_ID, "x")
        self.assertIsNone(record)
        self.assertIn("reserved", error)

    def test_chat_page_carries_the_token_only_when_auth_is_on(self):
        self.assertIn(INTERNAL_TOKEN, self.client.get("/chat").get_data(as_text=True))
        state.enable_auth = False
        page = self.client.get("/chat").get_data(as_text=True)
        self.assertNotIn(INTERNAL_TOKEN, page)
        self.assertIn('const INTERNAL_TOKEN = "";', page)


if __name__ == "__main__":
    unittest.main()
