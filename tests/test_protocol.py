import socket
import struct
import unittest

from moa.protocol import MAX_MESSAGE_BYTES, ProtocolError, recv_message, send_message


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = socket.socketpair()
        self.b.settimeout(2)

    def tearDown(self):
        self.a.close()
        self.b.close()

    def test_round_trip_keeps_unicode(self):
        message = {"v": 1, "type": "prompt", "prompt": "מה זה TCP? 🚀"}
        send_message(self.a, message)
        self.assertEqual(recv_message(self.b), message)

    def test_two_messages_sent_back_to_back_are_separated(self):
        send_message(self.a, {"n": 1})
        send_message(self.a, {"n": 2})
        self.assertEqual(recv_message(self.b), {"n": 1})
        self.assertEqual(recv_message(self.b), {"n": 2})

    def test_message_split_into_tiny_chunks_is_reassembled(self):
        payload = b'{"hello": "world"}'
        frame = struct.pack("!I", len(payload)) + payload
        for i in range(len(frame)):  # one byte per send()
            self.a.sendall(frame[i:i + 1])
        self.assertEqual(recv_message(self.b), {"hello": "world"})

    def test_connection_closed_mid_message(self):
        self.a.sendall(struct.pack("!I", 100) + b"{")
        self.a.close()
        with self.assertRaises(ProtocolError):
            recv_message(self.b)

    def test_oversized_length_prefix_is_rejected(self):
        self.a.sendall(struct.pack("!I", MAX_MESSAGE_BYTES + 1))
        with self.assertRaises(ProtocolError):
            recv_message(self.b)

    def test_non_object_json_is_rejected(self):
        payload = b"[1, 2, 3]"
        self.a.sendall(struct.pack("!I", len(payload)) + payload)
        with self.assertRaises(ProtocolError):
            recv_message(self.b)

    def test_invalid_json_is_rejected(self):
        payload = b"not json"
        self.a.sendall(struct.pack("!I", len(payload)) + payload)
        with self.assertRaises(ProtocolError):
            recv_message(self.b)


if __name__ == "__main__":
    unittest.main()
