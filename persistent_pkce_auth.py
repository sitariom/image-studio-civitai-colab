# -*- coding: utf-8 -*-
"""
Gerenciador de Autenticação PKCE Persistente para o Google Colab CLI.
Permite gerar uma URL de autenticação, salvar a verificação PKCE em disco (pkce_session.json)
e processar o código fornecido pelo usuário em chamadas separadas.
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
PKCE_STATE_PATH = os.path.expanduser("~/.config/colab-cli/pkce_session.json")
CLIENT_CONFIG_PATH = os.path.expanduser("~/.colab-cli-oauth-config.json")

def get_client_config():
    if os.path.exists(CLIENT_CONFIG_PATH):
        with open(CLIENT_CONFIG_PATH, "r") as f:
            return json.load(f)
    import importlib.resources as resources
    config_resource = resources.files("colab_cli").joinpath("oauth_config.json")
    return json.loads(config_resource.read_text())

def generate_new_url():
    client_config = get_client_config()
    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, state = flow.authorization_url(prompt="consent", token_usage="remote")
    
    # Persiste o estado PKCE em disco
    pkce_data = {
        "code_verifier": getattr(flow, "code_verifier", None),
        "state": state,
        "redirect_uri": REMOTE_REDIRECT_URI,
        "auth_url": auth_url
    }
    os.makedirs(os.path.dirname(PKCE_STATE_PATH), exist_ok=True)
    with open(PKCE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(pkce_data, f, indent=2)
        
    return auth_url

def exchange_code(code_str):
    code = code_str.strip()
    if not os.path.exists(PKCE_STATE_PATH):
        return False, "Sessão PKCE não encontrada. Por favor, gere uma nova URL primeiro."
    
    with open(PKCE_STATE_PATH, "r", encoding="utf-8") as f:
        pkce_data = json.load(f)
        
    code_verifier = pkce_data.get("code_verifier")
    if not code_verifier:
        return False, "code_verifier ausente no arquivo de sessão."
        
    client_config = get_client_config()
    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    flow.code_verifier = code_verifier
    
    try:
        flow.fetch_token(code=code)
        creds = flow.credentials
        
        os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
        with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
            
        return True, f"✅ Autenticação realizada com sucesso! Credenciais gravadas em {TOKEN_CONFIG_PATH}"
    except Exception as e:
        return False, f"Falha na troca de token: {e}"

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    
    args = sys.argv[1:]
    if not args or args[0] == "url":
        url = generate_new_url()
        print("URL_PKCE_PERSISTENTE:")
        print(url)
    elif args[0] == "auth" and len(args) >= 2:
        ok, msg = exchange_code(args[1])
        print(msg)
    else:
        print("Uso: persistent_pkce_auth.py [url | auth <code>]")
