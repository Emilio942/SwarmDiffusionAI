import os
import sys
import subprocess
import importlib.util
import argparse

# Liste der benötigten Pakete
REQUIRED_PACKAGES = [
    "numpy",
    "torch",
    "matplotlib"
]

def is_installed(package_name: str) -> bool:
    """
    Überprüft, ob ein Paket bereits installiert ist.
    
    :param package_name: Name des Pakets
    :return: True, wenn das Paket gefunden wurde, sonst False.
    """
    return importlib.util.find_spec(package_name) is not None

def install_package(package: str, upgrade: bool = False) -> None:
    """
    Installiert ein einzelnes Paket mithilfe von pip.
    
    :param package: Name des Pakets
    :param upgrade: Falls True, wird das Paket auch aktualisiert.
    """
    python_executable = sys.executable
    command = [python_executable, "-m", "pip", "install", package]
    if upgrade:
        command.append("--upgrade")
    print(f"📦 Installiere {package}...")
    try:
        subprocess.run(command, check=True)
        print(f"✅ {package} erfolgreich installiert.")
    except subprocess.CalledProcessError as error:
        print(f"❌ Fehler bei der Installation von {package}: {error}")
        sys.exit(1)

def install_packages(upgrade: bool = False) -> None:
    """
    Überprüft und installiert alle benötigten Pakete.
    Falls ein Paket bereits installiert ist, wird es übersprungen, außer ein Upgrade wird angefordert.
    
    :param upgrade: Falls True, werden alle Pakete auf die neueste Version aktualisiert.
    """
    for package in REQUIRED_PACKAGES:
        if is_installed(package):
            print(f"🔎 {package} ist bereits installiert.")
            if upgrade:
                print(f"⏫ Upgrade {package} wird durchgeführt...")
                install_package(package, upgrade=True)
        else:
            install_package(package, upgrade=upgrade)
    
    print("\n✅ Alle Pakete wurden erfolgreich überprüft und installiert!")

def main():
    parser = argparse.ArgumentParser(description="Installiere erforderliche Pakete für das Projekt.")
    parser.add_argument("--upgrade", action="store_true", help="Erzwingt das Upgrade aller Pakete.")
    args = parser.parse_args()
    
    install_packages(upgrade=args.upgrade)

if __name__ == "__main__":
    main()
