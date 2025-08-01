from colorama import Fore, Style, init

init(autoreset=True)

def printw(message):
    print(Fore.YELLOW + "[WARNING] " + message)

def printi(message):
    print(Fore.CYAN + "[INFO] " + message)

def printe(message):
    print(Fore.RED + "[ERROR] " + message)
