"""
FairSkin AI — One-Click Application Launcher.
Automatically launches the local offline server and opens the browser.
"""

import sys
import os
import webbrowser
import threading
import time

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from app import app


def open_browser():
    time.sleep(1.2)
    url = "http://127.0.0.1:5000"
    print(f"\nOpening screening interface in default browser: {url}...")
    webbrowser.open(url)


if __name__ == "__main__":
    print("=" * 65)
    print("   FairSkin AI — Offline Skin Disease Screening System")
    print("   Targets: Acne, Melasma, Psoriasis, Ringworm, Vitiligo")
    print("   Status: 100% Offline / Local Inference (No Cloud Calls)")
    print("=" * 65)

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
