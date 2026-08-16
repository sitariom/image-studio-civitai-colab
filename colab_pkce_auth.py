# -*- coding: utf-8 -*-
import os, sys, json
from google_auth_oauthlib.flow import InstalledAppFlow
import colab_cli.auth as ca

# Scopes validados pelo Google para este cliente OAuth (userinfo.profile e INVALIDO)
PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
REMOTE_REDIRECT_URI = ca.REMOTE_REDIRECT_URI
TOKEN_CONFIG_PATH = ca.TOKEN_CONFIG_PATH
VERIFIER_PATH = os.path.expanduser("~/.config/colab-cli-verifier.json")

config_resource = ca.resources.files('colab_cli').joinpath('oauth_config.json')
client_config = json.loads(config_resource.read_text())

if len(sys.argv) > 1 and sys.argv[1] == "url":
    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, state = flow.authorization_url(prompt='consent', token_usage='remote')
    
    print("\n🔗 URL Oficial de Autorização do Google Colab CLI (idêntica à do 'colab new'):\n")
    print(auth_url)
    print("\nAcesse a URL acima, clique em 'Permitir' e cole o codigo gerado!")

elif len(sys.argv) > 1 and sys.argv[1] == "submit":
    if len(sys.argv) < 3:
        print("Uso: python3 colab_pkce_auth.py submit <codigo>")
        sys.exit(1)
    
    code = sys.argv[2].strip()
    if not os.path.exists(VERIFIER_PATH):
        print("Erro: Gere a URL primeiro com 'python3 colab_pkce_auth.py url'")
        sys.exit(1)
        
    with open(VERIFIER_PATH, "r") as f:
        data = json.load(f)
        
    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    flow.code_verifier = data["code_verifier"]
    
    print(f"Enviando codigo {code[:10]}... com code_verifier sincronizado!")
    flow.fetch_token(code=code)
    creds = flow.credentials
    
    os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
    with open(TOKEN_CONFIG_PATH, 'w') as f:
        f.write(creds.to_json())
    print('\n✅ Sucesso absoluto! Credenciais salvas em:', TOKEN_CONFIG_PATH)
