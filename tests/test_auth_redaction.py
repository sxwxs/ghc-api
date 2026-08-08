import unittest

from ghc_api.auth import redact_auth_headers


class AuthHeaderRedactionTests(unittest.TestCase):
    def test_credentials_and_cookies_are_redacted_case_insensitively(self):
        headers = {
            "Authorization": "Bearer secret",
            "Proxy-Authorization": "Basic secret",
            "X-Api-Key": "secret",
            "Cookie": "session=secret",
            "Set-Cookie": "session=secret",
            "X-GitHub-Token": "secret",
            "Vendor-Access-Token": "secret",
            "Vendor-Secret": "secret",
            "User-Agent": "claude-cli/fixture",
        }

        redacted = redact_auth_headers(headers)

        self.assertEqual(redacted["User-Agent"], "claude-cli/fixture")
        for key in headers.keys() - {"User-Agent"}:
            self.assertEqual(redacted[key], "***REDACTED***", key)
        self.assertEqual(headers["Cookie"], "session=secret")


if __name__ == "__main__":
    unittest.main()
