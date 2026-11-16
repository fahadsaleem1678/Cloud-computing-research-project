"""HTTP suffix-inference service using the same checkpoint as the edge."""

from __future__ import annotations

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from split_model import load_checkpoint

MAX_REQUEST_BYTES = 64 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    name, model, digest = load_checkpoint(args.checkpoint, device)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.respond(404, b"not found", "text/plain")
                return
            payload = json.dumps({"model_name": name, "checkpoint_sha256": digest,
                                  "final_cut": model.final_cut, "device": str(device)}).encode()
            self.respond(200, payload, "application/json")

        def do_POST(self) -> None:
            if self.path != "/infer":
                self.respond(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                cut = int(self.headers.get("X-Cut", "-1"))
                if not 0 <= cut < model.final_cut or not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid cut or request length")
                activation = torch.load(io.BytesIO(self.rfile.read(length)), map_location="cpu", weights_only=True)
                if not isinstance(activation, torch.Tensor):
                    raise ValueError("Payload is not a tensor")
                with torch.inference_mode():
                    logits = model.forward_range(activation.to(device), cut, model.final_cut).cpu()
                buffer = io.BytesIO()
                torch.save(logits, buffer)
                self.respond(200, buffer.getvalue(), "application/octet-stream")
            except (ValueError, RuntimeError, EOFError) as error:
                self.respond(400, str(error).encode(), "text/plain")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {name} checkpoint={digest} at {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
