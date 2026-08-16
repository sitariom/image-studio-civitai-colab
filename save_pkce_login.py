# -*- coding: utf-8 -*-
"""
Gerenciador PKCE Persistente com Salvamento de code_verifier em Disco.
Garante que o code_verifier do PKCE sobreviva a reinícios do WSL / processos.
"""
import os
import sys
import json
import time
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
STATE_PATH = os.path.expanduser("~/.config/colab-cli/pkce_state.json")

CLIENT_CONFIG = {
    "installed": {
        "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
        "project_id": "colab-cli",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
        "redirect_uris": ["http://localhost"]
    }
}

def generate_url():
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote")
    
    verifier = None
    # InstalledAppFlow stores the verifier on the underlying oauthlib client
    oauth_client = getattr(flow, "oauth2session", None)
    if oauth_client is not None:
        verifier = getattr(oauth_client, "code_verifier", None)
    if verifier is None:
        verifier = getattr(flow, "code_verifier", None)
    
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"code_verifier": verifier, "auth_url": auth_url}, f)
    
    return auth_url

def exchange_code(code_str):
    if not os.path.exists(STATE_PATH):
        raise FileNotFoundError("Estado PKCE não encontrado. Gere uma nova URL.")
    
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state_data = json.load(f)
    
    verifier = state_data.get("code_verifier")
    
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    if verifier:
        # Set the verifier directly on the underlying oauthlib client
        oauth_client = getattr(flow, "oauth2session", None)
        if oauth_client is not None:
            oauth_client.code_verifier = verifier
        try:
            flow.code_verifier = verifier
        except Exception:
            pass
    
    flow.fetch_token(code=code_str.strip())
    creds = flow.credentials

    os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
    with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    
    return "✅ Token da nova conta salvo com sucesso!"

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "url":
        print("URL_GENERATED:")
        print(generate_url())
    elif args[0] == "exchange" and len(args) > 1:
        print(exchange_code(args[1]))
