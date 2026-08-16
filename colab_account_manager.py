# -*- coding: utf-8 -*-
"""
Google Colab Multi-Account GPU & Auth Lifecycle Manager
Gerencia o registro de autenticação de múltiplas contas do Google,
a rotação de contas quando o limite de GPU é atingido e solicita
reautenticação com links diretos sempre que necessário.
"""

import os
import json
import time
import subprocess
import re

APP_DIR = os.path.expanduser("~/.pi/agent/colab")
ACCOUNTS_FILE = os.path.join(APP_DIR, "colab_accounts.json")
COLAB_LOGIN_URL = "https://colab.research.google.com/"
GOOGLE_AUTH_URL = "https://accounts.google.com/"

os.makedirs(APP_DIR, exist_ok=True)

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

def record_authenticated_account(name, email="", auth_token="", note=""):
    """Grava e registra uma conta em que o usuário se autenticou."""
    data = load_accounts()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Procura se ja existe
    found = False
    for idx, acc in enumerate(data["accounts"]):
        if acc["name"].lower() == name.lower() or (email and acc.get("email", "").lower() == email.lower()):
            acc["name"] = name
            acc["email"] = email or acc.get("email", "")
            acc["auth_token"] = auth_token or acc.get("auth_token", "")
            acc["note"] = note or acc.get("note", "")
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
            "auth_token": auth_token,
            "note": note,
            "gpu_available": True,
            "limit_reached_at": None,
            "authenticated_at": now_str,
            "last_updated": now_str
        }
        data["accounts"].append(new_acc)
        data["active_index"] = len(data["accounts"]) - 1
    
    save_accounts(data)
    return f"✅ Conta '{name}' ({email or 'sem email'}) autenticada e gravada como ATIVA."

def remove_account(name):
    data = load_accounts()
    data["accounts"] = [acc for acc in data["accounts"] if acc["name"].lower() != name.lower()]
    if data["active_index"] >= len(data["accounts"]):
        data["active_index"] = max(0, len(data["accounts"]) - 1)
    save_accounts(data)
    return f"Conta '{name}' removida."

def get_active_account():
    data = load_accounts()
    accs = data.get("accounts", [])
    if not accs:
        return None
    idx = min(data.get("active_index", 0), len(accs) - 1)
    return accs[idx]

def set_active_account(name_or_idx):
    data = load_accounts()
    accs = data.get("accounts", [])
    if not accs:
        return "Nenhuma conta cadastrada."
    
    target_idx = None
    if isinstance(name_or_idx, int) or str(name_or_idx).isdigit():
        target_idx = int(name_or_idx)
        if target_idx < 0 or target_idx >= len(accs):
            return f"Índice {target_idx} inválido. Total de contas: {len(accs)}"
    else:
        for idx, acc in enumerate(accs):
            if acc["name"].lower() == str(name_or_idx).lower() or acc.get("email", "").lower() == str(name_or_idx).lower():
                target_idx = idx
                break
    
    if target_idx is None:
        return f"Conta '{name_or_idx}' não encontrada."
    
    data["active_index"] = target_idx
    save_accounts(data)
    active = accs[target_idx]
    return f"🔄 Conta ativa alterada para: {active['name']} ({active.get('email', 'sem email')})"

def mark_gpu_limit_reached(account_name=None):
    data = load_accounts()
    accs = data.get("accounts", [])
    if not accs:
        return "Nenhuma conta para marcar."
    
    target_acc = None
    if account_name:
        for acc in accs:
            if acc["name"].lower() == account_name.lower():
                target_acc = acc
                break
    else:
        idx = min(data.get("active_index", 0), len(accs) - 1)
        target_acc = accs[idx]
    
    if target_acc:
        target_acc["gpu_available"] = False
        target_acc["limit_reached_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_accounts(data)
        return f"Limite de GPU registrado para a conta '{target_acc['name']}'."
    return "Conta não encontrada."

def request_reauthentication(account_name=None, reason="Autenticação expirada ou limite de GPU atingido em todas as contas."):
    """Gera uma solicitação formal de reautenticação com links diretos."""
    target_info = f" para a conta **{account_name}**" if account_name else ""
    msg = (
        f"🔐 **Solicitação de Reautenticação no Google Colab**{target_info}\n\n"
        f"**Motivo**: {reason}\n\n"
        f"Por favor, realize o login através de um dos links abaixo para renovar a sessão de GPU:\n"
        f"1. 🌐 **Google Colab**: [{COLAB_LOGIN_URL}]({COLAB_LOGIN_URL})\n"
        f"2. 🔑 **Login Google**: [{GOOGLE_AUTH_URL}]({GOOGLE_AUTH_URL})\n\n"
        f"Assim que concluir a autenticação, me avise para que eu grave a nova sessão e retome a execução!"
    )
    return msg

def auto_failover_next_available():
    """Alterna para a próxima conta gravada. Se todas estiverem sem GPU, solicita reautenticação."""
    data = load_accounts()
    accs = data.get("accounts", [])
    if not accs:
        return None, request_reauthentication(reason="Nenhuma conta Google cadastrada. Por favor, autentique a primeira conta.")
    
    current_idx = data.get("active_index", 0)
    current_name = "Desconhecida"
    if current_idx < len(accs):
        accs[current_idx]["gpu_available"] = False
        accs[current_idx]["limit_reached_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        current_name = accs[current_idx]["name"]
    
    # Procura a proxima conta com GPU disponivel
    for i in range(1, len(accs)):
        next_idx = (current_idx + i) % len(accs)
        if accs[next_idx].get("gpu_available", True):
            data["active_index"] = next_idx
            save_accounts(data)
            next_acc = accs[next_idx]
            return next_acc, f"🔄 Failover executado: Limite de GPU atingido na conta '{current_name}'. Migrado para a conta gravada '{next_acc['name']}' ({next_acc.get('email', 'sem email')})!"
    
    # Todas as contas gravadas atingiram o limite -> solicita reautenticacao
    save_accounts(data)
    reauth_msg = request_reauthentication(reason=f"Todas as {len(accs)} contas salvas atingiram o limite diário de GPU do Colab.")
    return None, reauth_msg

def get_status_summary():
    data = load_accounts()
    accs = data.get("accounts", [])
    active_idx = data.get("active_index", 0)
    
    lines = ["### 📊 Registro de Contas Google Colab Autenticadas\n"]
    if not accs:
        lines.append("Nenhuma conta gravada. Quando você fizer login, me avise para gravar a conta.")
        return "\n".join(lines)
    
    for idx, acc in enumerate(accs):
        is_active = (idx == active_idx)
        prefix = "👉 **[ATIVA]** " if is_active else "   "
        gpu_status = "✅ GPU Disponível" if acc.get("gpu_available", True) else f"❌ Limite Atingido ({acc.get('limit_reached_at', '-')})"
        auth_time = acc.get("authenticated_at", "Não registrada")
        lines.append(f"{prefix}{idx+1}. **{acc['name']}** ({acc.get('email', 'sem email')}) — {gpu_status} | *Autenticada em*: {auth_time}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    args = sys.argv[1:]
    if not args or args[0] in ("status", "list"):
        print(get_status_summary())
    elif args[0] in ("record", "add") and len(args) >= 2:
        name = args[1]
        email = args[2] if len(args) > 2 else ""
        print(record_authenticated_account(name, email))
    elif args[0] == "switch" and len(args) >= 2:
        print(set_active_account(args[1]))
    elif args[0] == "failover":
        acc, msg = auto_failover_next_available()
        print(msg)
    elif args[0] == "reauth":
        name = args[1] if len(args) > 1 else None
        print(request_reauthentication(name))
    else:
        print("Uso: colab_account_manager.py [status|record <nome> <email>|switch <nome_ou_idx>|failover|reauth]")
