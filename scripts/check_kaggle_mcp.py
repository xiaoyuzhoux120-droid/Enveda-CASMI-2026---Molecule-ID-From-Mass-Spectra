"""Read-only probe of the official Kaggle MCP server (https://www.kaggle.com/mcp).

Runs the MCP handshake and lists tools. If KAGGLE_API_TOKEN is set it is sent
as a Bearer header (never printed). It never calls any tool, so it cannot
submit, publish or change anything.
"""
import json
import os
import sys

import requests

URL = "https://www.kaggle.com/mcp"
PROTOCOL = "2025-03-26"  # the server answered 404 to 2025-06-18 during setup


def parse(resp):
    text = resp.text
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:])
    return json.loads(text) if text.strip() else {}


def main():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json",
                      "Accept": "application/json, text/event-stream"})
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    r = s.post(URL, timeout=30, json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": PROTOCOL, "capabilities": {},
                          "clientInfo": {"name": "casmi-env-check", "version": "1"}}})
    print("initialize:", r.status_code, parse(r).get("result", {}).get("serverInfo"))
    if r.status_code != 200:
        sys.exit(1)
    sid = r.headers.get("Mcp-Session-Id")
    if sid:
        s.headers["Mcp-Session-Id"] = sid
    s.headers["MCP-Protocol-Version"] = PROTOCOL
    s.post(URL, timeout=30, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    r = s.post(URL, timeout=60, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = parse(r).get("result", {}).get("tools", [])
    print("tools/list:", r.status_code, "count =", len(tools), "| auth =", bool(token))
    for t in tools:
        print(" -", t["name"])


if __name__ == "__main__":
    main()
