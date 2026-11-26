import os
import torch


def cargar_texto(archivos):
    textos = []

    for archivo in archivos:
        if not os.path.exists(archivo):
            print(f"Aviso: no se encontró {archivo}, se omite.")
            continue

        with open(archivo, "r", encoding="utf-8") as f:
            textos.append(f.read())

    if len(textos) == 0:
        raise FileNotFoundError("No se encontró ningún archivo de entrenamiento.")

    return "\n\n".join(textos)


def crear_vocabulario(texto):
    chars = sorted(list(set(texto)))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return chars, stoi, itos


def encode(texto, stoi):
    tokens = []

    for c in texto:
        if c in stoi:
            tokens.append(stoi[c])
        elif " " in stoi:
            tokens.append(stoi[" "])

    return tokens


def decode(tokens, itos):
    return "".join([itos[i] for i in tokens])


def guardar_checkpoint(path, model, stoi, itos, config):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": config,
    }

    torch.save(checkpoint, path)


def cargar_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró {path}. Primero entrena Sancho v0.4 ejecutando: python train_v0_4.py"
        )

    return torch.load(path, map_location=device)