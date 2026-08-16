# -*- coding: utf-8 -*-
"""
Script de Login Direto para o Google Colab CLI.
Salva o token em ~/.config/colab-cli/token.json de forma 100% compatível.
"""
import os
import sys
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]

REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
TOKEN_CONFIG_PATH = os.path.expanduser("~/.config/colab-cli/token.json")

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    config_path = os.path.expanduser("~/.colab-cli-oauth-config.json")
    if not os.path.exists(config_path):
        # Default Google Colab CLI OAuth Config
        client_config = {
            "installed": {
                "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
                "project_id": "google.com:cloudsdk",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs"
            }
        }
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            client_config = json.load(f)

    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote")

    print("\n" + "="*70)
    print("URL DE LOGIN PARA A NOVA CONTA:")
    print(auth_url)
    print("="*70 + "\n", flush=True)

    with open("pending_url.txt", "w", encoding="utf-8") as f:
        f.write(auth_url)

    print("Aguardando código em input_code.txt...", flush=True)
    code = None
    for _ in range(300):
        if os.path.exists("input_code.txt"):
            code = open("input_code.txt", encoding="utf-8").read().strip()
            if code:
                os.remove("input_code.txt")
                break
        time.sleep(1)

    if not code:
        print("TIMEOUT: Nenhum código fornecido.")
        sys.exit(1)

    print(f"Trocando código pelo token de acesso com o mesmo flow...", flush=True)
    flow.fetch_token(code=code)
    creds = flow.credentials

    os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
    with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    print("✅ TOKEN DA NOVA CONTA SALVO COM SUCESSO EM:", TOKEN_CONFIG_PATH)

if __name__ == "__main__":
    import time
    main()
