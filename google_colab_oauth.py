# -*- coding: utf-8 -*-
"""
Google OAuth 2.0 Desktop Authentication Flow para Colab CLI.
Permite autenticar contas reais do Google em segundo plano,
armazenando tokens OAuth para rotação automática de GPU.
"""

import os
import sys
import json
import time
import urllib.request
import requests
from pathlib import Path

APP_DIR = os.path.expanduser("~/.pi/agent/colab")
ACCOUNTS_FILE = os.path.join(APP_DIR, "colab_accounts.json")
CREDENTIALS_FILE = os.path.join(APP_DIR, "google_credentials.json")
os.makedirs(APP_DIR, exist_ok=True)

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/drive.file",
    "openid"
]

def load_accounts():
    if os.path.exists(ACCOUNTS_FILE):
        try:
            with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"accounts": [], "active_index": 0}

def save_accounts(data):
    with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def fetch_user_email(access_token):
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        r = requests.get("https://www.googleapis.com/oauth2/v2/userinfo", headers=headers, timeout=20)
        if r.status_code == 200:
            return r.json().get("email", "")
    except Exception:
        pass
    return ""

def register_authenticated_tokens(access_token, refresh_token=None, account_label=None):
    email = fetch_user_email(access_token) or "conta_google@gmail.com"
    name = account_label or f"Conta_{email.split('@')[0]}"

    data = load_accounts()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    found = False
    for idx, acc in enumerate(data["accounts"]):
        if acc.get("email", "").lower() == email.lower() or acc["name"].lower() == name.lower():
            acc["name"] = name
            acc["email"] = email
            acc["access_token"] = access_token
            if refresh_token:
                acc["refresh_token"] = refresh_token
            acc["gpu_available"] = True
            acc["limit_reached_at"] = None
            acc["authenticated_at"] = now_str
            acc["last_updated"] = now_str
            data["active_index"] = idx
            found = True
            break

    if not found:
        new_acc = {
            "name": name,
            "email": email,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "gpu_available": True,
            "limit_reached_at": None,
            "authenticated_at": now_str,
            "last_updated": now_str
        }
        data["accounts"].append(new_acc)
        data["active_index"] = len(data["accounts"]) - 1

    save_accounts(data)
    return name, email

def find_credentials_file():
    if os.path.exists(CREDENTIALS_FILE):
        return CREDENTIALS_FILE
    if os.path.exists(APP_DIR):
        for f in os.listdir(APP_DIR):
            if (f.startswith("client_secret") or "credential" in f.lower()) and f.endswith(".json") and f != "colab_accounts.json":
                return os.path.join(APP_DIR, f)
    return None

def run_local_oauth_flow(client_secrets_path=None, account_label=None):
    """Executa o fluxo de autenticacao OAuth 2.0 Desktop do Google."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        raise RuntimeError("Biblioteca 'google-auth-oauthlib' não encontrada. Instale com: pip install google-auth-oauthlib")

    if not client_secrets_path or not os.path.exists(client_secrets_path):
        client_secrets_path = find_credentials_file()

    if not client_secrets_path or not os.path.exists(client_secrets_path):
        return None, (
            "⚠️ Arquivo de credenciais 'google_credentials.json' não encontrado.\n"
            "Para autenticar no modo CLI remoto do Google:\n"
            "1. Baixe o arquivo de credenciais OAuth 2.0 no Google Cloud Console.\n"
            f"2. Salve o arquivo em `{APP_DIR}`.\n"
            "3. Execute `python google_colab_oauth.py login`."
        )

    flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, scopes=SCOPES)
    try:
        auth_url, _ = flow.authorization_url(prompt="consent")
        print(f"🔗 URL de Autenticação Oficial do Google:\n{auth_url}\n")
        print("Aguardando confirmação de login no seu navegador...")
    except Exception:
        pass
    creds = flow.run_local_server(port=0)

    access_token = creds.token
    refresh_token = creds.refresh_token
    name, email = register_authenticated_tokens(access_token, refresh_token, account_label)
    return True, f"✅ Autenticação realizada com sucesso para a conta '{name}' ({email})!"

def create_credentials_json(client_id, client_secret):
    """Cria o arquivo google_credentials.json a partir do client_id e client_secret."""
    content = {
        "installed": {
            "client_id": client_id.strip(),
            "project_id": "colab-studio-app",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret.strip(),
            "redirect_uris": ["http://localhost"]
        }
    }
    with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
    return f"✅ Credenciais registradas com sucesso em '{CREDENTIALS_FILE}'!"

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    args = sys.argv[1:]
    if args and args[0] == "login":
        cred_file = args[1] if len(args) > 1 else None
        label = args[2] if len(args) > 2 else None
        ok, msg = run_local_oauth_flow(cred_file, label)
        print(msg)
    elif args and args[0] == "keys" and len(args) >= 3:
        cid = args[1]
        csec = args[2]
        print(create_credentials_json(cid, csec))
    elif args and args[0] == "register" and len(args) >= 2:
        acc_token = args[1]
        ref_token = args[2] if len(args) > 2 else None
        label = args[3] if len(args) > 3 else None
        name, email = register_authenticated_tokens(acc_token, ref_token, label)
        print(f"✅ Conta '{name}' ({email}) registrada no gerenciador.")
    else:
        print("Uso: google_colab_oauth.py [login [caminho_json] | keys <client_id> <client_secret> | register <access_token>]")
