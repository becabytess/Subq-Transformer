#!/usr/bin/env python3
"""
Simple HTTP server to run the SubQ 3D Receptive Field Visualizer.
Usage:
    python serve.py [port]
"""

import http.server
import socketserver
import os
import sys
import webbrowser

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler

# Allow immediate reuse of address
socketserver.TCPServer.allow_reuse_address = True

print("=" * 65)
print("SubQ 3D Receptive Field Explorer Server")
print(f"Local URL: http://localhost:{PORT}")
print("=" * 65)
print("Opening browser...")

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        webbrowser.open(f"http://localhost:{PORT}")
        print("Server running. Press Ctrl+C to stop.")
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
except Exception as e:
    print(f"Error starting server: {e}")
