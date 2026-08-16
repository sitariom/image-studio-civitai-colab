# -*- coding: utf-8 -*-
import os, sys, json, time
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
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
CLIENT_CONFIG_PATH = os.path.expanduser("~/.colab-cli-oauth-config.json")

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    if not os.path.exists(CLIENT_CONFIG_PATH):
        # try inlined resource
        import importlib.resources as resources
        try:
            config_resource = resources.files("colab_cli").joinpath("oauth_config.json")
            if config_resource.is_file():
                client_config = json.loads(config_resource.read_text())
        except Exception as e:
            print("Error loading config:", e)
            sys.exit(1)
    else:
        with open(CLIENT_CONFIG_PATH, "r") as f:
            client_config = json.load(f)

    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote")

    with open("current_auth_url.txt", "w", encoding="utf-8") as f:
        f.write(auth_url)

    print("="*80, flush=True)
    print("URL_AUTENTICACAO_ATIVA:", flush=True)
    print(auth_url, flush=True)
    print("="*80, flush=True)

    code = None
    if os.path.exists("code_input.txt"):
        os.remove("code_input.txt")

    print("Aguardando codigo em code_input.txt...", flush=True)
    for _ in range(180):
        if os.path.exists("code_input.txt"):
            c = open("code_input.txt", encoding="utf-8").read().strip()
            if c:
                code = c
                break
        time.sleep(1)

    if not code:
        print("ERR: Timeout sem codigo.", flush=True)
        sys.exit(1)

    print(f"Processando codigo ({len(code)} chars)...", flush=True)
    flow.fetch_token(code=code)
    creds = flow.credentials

    os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
    with open(TOKEN_CONFIG_PATH, "w") as f:
        f.write(creds.to_json())

    print("SUCESSO: Credenciais salvas em", TOKEN_CONFIG_PATH, flush=True)

if __name__ == "__main__":
    main()
