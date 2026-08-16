# -*- coding: utf-8 -*-
import os
import sys
import time
import json
from google_auth_oauthlib.flow import InstalledAppFlow

PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
TOKEN_CONFIG_PATH = os.path.expanduser("~/.config/colab-cli/token.json")
BASE_DIR = "/mnt/c/Users/simoe/Downloads/image_gerador_colab"

def main():
    code_path = os.path.join(BASE_DIR, "auth_code.txt")
    url_path = os.path.join(BASE_DIR, "auth_url.txt")

    if os.path.exists(code_path):
        try: os.remove(code_path)
        except: pass
    if os.path.exists(url_path):
        try: os.remove(url_path)
        except: pass

    client_config = {
        "installed": {
            "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
            "project_id": "google.com:cloudsdk",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs"
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote")

    with open(url_path, "w", encoding="utf-8") as f:
        f.write(auth_url)

    print("URL_READY:", auth_url, flush=True)

    code = None
    for _ in range(600):
        if os.path.exists(code_path):
            code = open(code_path, encoding="utf-8").read().strip()
            if code:
                os.remove(code_path)
                break
        time.sleep(0.5)

    if not code:
        print("FAIL: Timeout aguardando código", flush=True)
        sys.exit(1)

    print(f"Exchanging code {code[:10]}...", flush=True)
    flow.fetch_token(code=code)
    creds = flow.credentials

    os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
    with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    print("SUCCESS: Token salvo em", TOKEN_CONFIG_PATH, flush=True)

if __name__ == "__main__":
    main()
