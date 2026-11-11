"""The Gmail MCP server against a fake Gmail service (no Google account needed).

FakeGmail answers the same calls as the real client library
(`service.users().messages().list(...).execute()`), from an in-memory mailbox.
"""

import asyncio
import base64
import tempfile
import unittest
from pathlib import Path

import httplib2
from googleapiclient.errors import HttpError
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from moa.servers.gmail import build_server, load_credentials
from moa.servers.gmail_box import MAX_BODY_CHARS, MAX_SEARCH_RESULTS, UNTRUSTED_START, GmailBox


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode().rstrip("=")


def _message(mid, sender, subject, body=None, html=None, date="Mon, 9 Nov 2026 10:00:00 +0200"):
    headers = [{"name": "From", "value": sender}, {"name": "To", "value": "me@example.com"},
               {"name": "Subject", "value": subject}, {"name": "Date", "value": date}]
    parts = []
    if body is not None:
        parts.append({"mimeType": "text/plain", "body": {"data": _b64(body)}})
    if html is not None:
        parts.append({"mimeType": "text/html", "body": {"data": _b64(html)}})
    return {"id": mid, "snippet": (body or "")[:40],
            "payload": {"mimeType": "multipart/alternative", "headers": headers, "parts": parts}}


class _Request:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def execute(self):
        if self.error:
            raise self.error
        return self.result


class FakeGmail:
    def __init__(self, messages):
        self.messages_by_id = {m["id"]: m for m in messages}
        self.trashed = []
        self.calls = []

    def users(self):
        return self

    def messages(self):
        return self

    def list(self, userId, q, maxResults):
        self.calls.append(("list", q, maxResults))
        hits = [m for m in self.messages_by_id.values()
                if not q or q.lower() in str(m["payload"]["headers"]).lower()]
        return _Request({"messages": [{"id": m["id"]} for m in hits[:maxResults]]} if hits else {})

    def get(self, userId, id, format, metadataHeaders=None):
        self.calls.append(("get", id, format))
        if id not in self.messages_by_id:
            return _Request(error=HttpError(httplib2.Response({"status": 404}), b"not found"))
        return _Request(self.messages_by_id[id])

    def trash(self, userId, id):
        self.calls.append(("trash", id))
        if id not in self.messages_by_id:
            return _Request(error=HttpError(httplib2.Response({"status": 404}), b"not found"))
        self.trashed.append(id)
        return _Request({"id": id, "labelIds": ["TRASH"]})


MAILBOX = [
    _message("a1", "Dana <dana@example.com>", "Meeting on Thursday", "Lab meeting Thursday 12/11 at 14:00."),
    _message("b2", "Boss <boss@example.com>", "Task: report", "Please send the report by Sunday."),
    _message("c3", "News <news@example.com>", "Weekly digest", html="<p>Hello&nbsp;<b>world</b></p><script>x()</script>"),
    _message("d4", "שירה <shira@example.com>", "תזכורת", "פגישה מחר ב-10"),
]


class GmailBoxTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeGmail(MAILBOX)
        self.box = GmailBox(self.fake)

    def test_search_lists_id_sender_subject_and_snippet(self):
        text = self.box.search_emails("meeting")
        self.assertIn("id=a1", text)
        self.assertIn("from: Dana <dana@example.com>", text)
        self.assertIn("subject: Meeting on Thursday", text)
        self.assertIn("Lab meeting", text)
        self.assertNotIn("id=b2", text)

    def test_search_is_capped(self):
        self.box.search_emails("", max_results=500)
        self.assertEqual(self.fake.calls[0], ("list", "", MAX_SEARCH_RESULTS))

    def test_search_with_no_match_says_so(self):
        self.assertIn("no emails match", self.box.search_emails("nothing-like-this"))

    def test_read_returns_headers_and_the_plain_text_body(self):
        text = self.box.read_email("b2")
        self.assertIn("From: Boss <boss@example.com>", text)
        self.assertIn("Subject: Task: report", text)
        self.assertIn("Please send the report by Sunday.", text)

    def test_body_is_marked_as_untrusted(self):
        text = self.box.read_email("a1")
        self.assertLess(text.index(UNTRUSTED_START), text.index("Lab meeting"))

    def test_hebrew_survives_decoding(self):
        self.assertIn("פגישה מחר ב-10", self.box.read_email("d4"))

    def test_html_only_email_is_turned_into_text(self):
        text = self.box.read_email("c3")
        self.assertIn("Hello world", text)
        self.assertNotIn("<b>", text)
        self.assertNotIn("x()", text)  # script content dropped

    def test_long_body_is_cut(self):
        self.fake.messages_by_id["e5"] = _message("e5", "x@example.com", "long", "a" * (MAX_BODY_CHARS + 500))
        text = self.box.read_email("e5")
        self.assertIn(f"cut at {MAX_BODY_CHARS} characters", text)
        self.assertLess(len(text), MAX_BODY_CHARS + 1000)

    def test_invalid_id_is_refused_before_any_api_call(self):
        with self.assertRaises(ToolError):
            self.box.read_email("../../users/other")
        self.assertEqual(self.fake.calls, [])

    def test_unknown_id_is_a_clear_error(self):
        with self.assertRaisesRegex(ToolError, "no email with id 'zz9'"):
            self.box.read_email("zz9")


class TrashTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeGmail(MAILBOX)
        self.box = GmailBox(self.fake)

    def test_trash_moves_the_email_and_says_it_can_be_restored(self):
        text = self.box.trash_email("b2")
        self.assertEqual(self.fake.trashed, ["b2"])
        self.assertIn("Trash", text)
        self.assertIn("restored", text)

    def test_trash_refuses_an_invalid_id_before_any_api_call(self):
        with self.assertRaises(ToolError):
            self.box.trash_email("me/../other")
        self.assertEqual(self.fake.calls, [])

    def test_trash_of_an_unknown_id_is_a_clear_error(self):
        with self.assertRaisesRegex(ToolError, "no email with id 'zz9'"):
            self.box.trash_email("zz9")


class GmailServerTests(unittest.TestCase):
    """The same box behind a real MCP server, through an in-memory MCP client."""

    def _with_client(self, action):
        async def run():
            async with Client(build_server(GmailBox(FakeGmail(MAILBOX)))) as client:
                return await action(client)
        return asyncio.run(run())

    def test_only_trash_is_marked_as_changing_the_mailbox(self):
        tools = {t.name: t for t in self._with_client(lambda c: c.list_tools()).tools}
        self.assertEqual(set(tools), {"search_emails", "read_email", "trash_email"})
        self.assertTrue(tools["search_emails"].annotations.read_only_hint)
        self.assertTrue(tools["read_email"].annotations.read_only_hint)
        self.assertFalse(tools["trash_email"].annotations.read_only_hint)
        self.assertTrue(tools["trash_email"].annotations.destructive_hint)
        for t in tools.values():
            self.assertTrue(t.description)

    def test_tool_call_through_mcp(self):
        result = self._with_client(lambda c: c.call_tool("read_email", {"message_id": "a1"}))
        self.assertFalse(result.is_error)
        self.assertIn("Lab meeting", result.content[0].text)

    def test_tool_error_reaches_the_client_as_an_error_result(self):
        result = self._with_client(lambda c: c.call_tool("read_email", {"message_id": "zz9"}))
        self.assertTrue(result.is_error)
        self.assertIn("no email with id", result.content[0].text)


class CredentialsTests(unittest.TestCase):
    def test_missing_token_tells_to_log_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "--login"):
                load_credentials(Path(tmp) / "token.json")


if __name__ == "__main__":
    unittest.main()
