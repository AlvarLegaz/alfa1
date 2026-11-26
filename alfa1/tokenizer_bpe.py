# tokenizer_bpe.py
import os
import json

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

try:
    from tqdm import tqdm
    _HAY_TQDM = True
except ImportError:
    _HAY_TQDM = False


def _contar_lineas_fichero(ruta):
    """A nivel de modulo (no metodo) para que multiprocessing pueda
    usarla como worker tambien en Windows, donde no hay fork y los
    workers se lanzan reimportando el modulo (spawn)."""
    n = 0
    with open(ruta, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


class BPETokenizer:
    """BPE sobre HF tokenizers. Version 1B: vocab 48k, digitos partidos,
    set de tokens especiales CONGELADO (no cambiar entre v1 y v2).

    Orden de especiales = IDs estables. NO reordenar.
        0 <PAD>  1 <UNK>  2 <BOS>  3 <EOS>  4 <USER>  5 <ASSISTANT>
        6 <FIN>  7 <tool>  8 </tool>  9 <turno>  10 <|endofstory|>
    """

    def __init__(self):
        self.pad_token = "<PAD>"; self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"; self.eos_token = "<EOS>"
        self.user_token = "<USER>"; self.assistant_token = "<ASSISTANT>"
        self.fin_token = "<FIN>"
        self.tool_token = "<tool>"; self.tool_end_token = "</tool>"
        self.turno_token = "<turno>"
        self.story_token = "<|endofstory|>"
        # El ORDEN fija los IDs. No tocar.
        self.special_tokens = [
            self.pad_token, self.unk_token, self.bos_token, self.eos_token,
            self.user_token, self.assistant_token, self.fin_token,
            self.tool_token, self.tool_end_token, self.turno_token,
            self.story_token,
        ]
        self._tok = None

    # --- Normalizacion de chat (igual que tu version) ---
    def normalize_chat_text(self, text, add_fin_if_missing=False):
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                lines.append(""); continue
            if line.startswith("Usuario:"):
                lines.append(f"{self.user_token} {line[len('Usuario:'):].strip()}")
            elif line.startswith("Sancho:"):
                c = line[len('Sancho:'):].strip()
                if add_fin_if_missing and self.fin_token not in c:
                    c = f"{c} {self.fin_token}"
                lines.append(f"{self.assistant_token} {c}")
            else:
                lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    def _build_trainer(self, vocab_size, min_pair_freq):
        return trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_pair_freq,
            special_tokens=self.special_tokens,
        )

    def _build_tokenizer(self):
        tok = Tokenizer(models.BPE(unk_token=self.unk_token))
        # Pre-tokenizador en SECUENCIA:
        #  1) Digits(individual=True): cada cifra es su propio token. Evita
        #     que el BPE fusione "115200", "9600", "2026" en un unico token
        #     y mejora el razonamiento numerico (baudrates, medidas, codigo).
        #  2) ByteLevel: cobertura total de bytes, sin <UNK> en la practica.
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ])
        tok.decoder = decoders.ByteLevel()
        return tok

    def _finalizar(self, tok):
        # add_special_tokens los registra como AddedToken para que
        # skip_special_tokens los filtre y el matching al encodear sea exacto.
        tok.add_special_tokens(self.special_tokens)
        self._tok = tok

    # ------------------------------------------------------------------
    # Entrenamiento en memoria (corpus ya cargado en un string)
    # ------------------------------------------------------------------
    def train(self, text, vocab_size=48000, min_pair_freq=2,
              add_fin_if_missing=True, normalizar_chat=True, chunk=2000):
        if normalizar_chat:
            text = self.normalize_chat_text(text, add_fin_if_missing=add_fin_if_missing)

        tok = self._build_tokenizer()
        trainer = self._build_trainer(vocab_size, min_pair_freq)

        lineas = text.splitlines()
        total_chunks = (len(lineas) + chunk - 1) // chunk

        def iter_lineas():
            rango = range(0, len(lineas), chunk)
            if _HAY_TQDM:
                rango = tqdm(rango, total=total_chunks,
                             desc="Tokenizer (lineas)", unit="chunk")
            for i in rango:
                yield "\n".join(lineas[i:i + chunk])

        tok.train_from_iterator(iter_lineas(), trainer, length=total_chunks)
        self._finalizar(tok)
        return self.get_stoi(), None

    # ------------------------------------------------------------------
    # Entrenamiento desde los BLOQUES de mezclar_corpus.py (train/*.txt)
    # ------------------------------------------------------------------
    def train_desde_bloques(self, carpeta, vocab_size=48000, min_pair_freq=2,
                            patron="train_*.txt", max_ficheros=None, workers=None):
        """Entrena directamente sobre los bloques que produce
        mezclar_corpus.py en `salida/train/` (train_0000.txt, ...).

        Esos bloques ya vienen barajados GLOBALMENTE por el pipeline de
        mezcla (idiomas y temas revueltos entre si, no por bloque) asi
        que no hace falta volver a barajar ni hacer un muestreo aleatorio
        para entrenar el tokenizer con solo una fraccion del corpus:
        basta con coger los primeros `max_ficheros` bloques en orden, que
        ya son una muestra representativa del conjunto completo. Esto
        evita tener que leer los ~280 GB enteros solo para el BPE, que en
        la practica no necesita tanto texto para converger.

        Usa `max_ficheros=None` para entrenar sobre todos los bloques.
        """
        import glob
        rutas = sorted(glob.glob(os.path.join(carpeta, patron)))
        if not rutas:
            raise FileNotFoundError(
                f"No se ha encontrado ningun fichero '{patron}' en {carpeta}")

        total_disponibles = len(rutas)
        if max_ficheros is not None:
            rutas = rutas[:max_ficheros]

        tam_mb = sum(os.path.getsize(r) for r in rutas) / (1024 * 1024)
        print(f"[Tokenizer] Entrenando sobre {len(rutas)}/{total_disponibles} "
              f"bloques de '{carpeta}' ({tam_mb:.1f} MB)")
        if tam_mb > 15_000:
            print(f"[Tokenizer] AVISO: {tam_mb/1024:.1f} GB es bastante para "
                  f"entrenar un BPE -- el entrenador de HF tokenizers "
                  f"acumula la tabla de frecuencias de TODO el corpus en "
                  f"RAM (no es streaming real por dentro). Si te quedas sin "
                  f"memoria, baja --max-ficheros y/o sube --min-pair-freq "
                  f"(p.ej. 5-10 en vez de 2).")

        return self._entrenar_desde_iterador_lineas(
            rutas, vocab_size=vocab_size, min_pair_freq=min_pair_freq, workers=workers)

    # ------------------------------------------------------------------
    # Entrenamiento desde UN fichero (RAM plana)
    # ------------------------------------------------------------------
    def train_desde_fichero(self, ruta, vocab_size=48000, min_pair_freq=2):
        return self._entrenar_desde_iterador_lineas(
            [ruta], vocab_size=vocab_size, min_pair_freq=min_pair_freq
        )

    # ------------------------------------------------------------------
    # Entrenamiento desde VARIOS ficheros (RAM plana)
    # ------------------------------------------------------------------
    def _entrenar_desde_iterador_lineas(self, rutas, vocab_size=48000,
                                        min_pair_freq=2, workers=None):
        if isinstance(rutas, (str, os.PathLike)):
            rutas = [rutas]

        tok = self._build_tokenizer()
        trainer = self._build_trainer(vocab_size, min_pair_freq)

        # El conteo previo de lineas (solo para el total de la barra de
        # progreso) es I/O secuencial en Python puro: con muchos ficheros
        # de varios cientos de MB cada uno, merece la pena repartirlo entre
        # procesos en vez de recorrerlo fichero a fichero.
        n_workers = workers or min(len(rutas), os.cpu_count() or 1)
        if n_workers > 1 and len(rutas) > 1:
            import multiprocessing as mp
            with mp.Pool(processes=n_workers) as pool:
                conteos = pool.map(_contar_lineas_fichero, rutas)
            total_lineas = sum(conteos)
        else:
            total_lineas = sum(_contar_lineas_fichero(r) for r in rutas)

        tam_mb = sum(os.path.getsize(r) for r in rutas) / (1024 * 1024)
        print(f"Entrenando tokenizer desde {len(rutas)} fichero(s) "
              f"({tam_mb:.1f} MB, {total_lineas:,} lineas, vocab={vocab_size})...")

        def iter_lineas():
            for ruta in rutas:
                with open(ruta, "r", encoding="utf-8", errors="replace") as f:
                    for linea in f:
                        yield linea

        it = iter_lineas()
        if _HAY_TQDM:
            it = tqdm(it, total=total_lineas,
                      desc="Tokenizer (lineas)", unit="linea")

        tok.train_from_iterator(it, trainer, length=total_lineas)
        self._finalizar(tok)
        print(f"Tokenizer entrenado. Vocab real: {self.vocab_size()} tokens.")
        return self.get_stoi(), None

    # ------------------------------------------------------------------
    # API de uso
    # ------------------------------------------------------------------
    def encode(self, text, normalize_chat=True, add_bos=False, add_eos=False):
        if normalize_chat:
            text = self.normalize_chat_text(text, add_fin_if_missing=False)
        ids = self._tok.encode(text).ids
        if add_bos: ids = [self.bos_id] + ids
        if add_eos: ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts, normalize_chat=True, add_bos=False, add_eos=False):
        if normalize_chat:
            texts = [self.normalize_chat_text(t, add_fin_if_missing=False) for t in texts]
        batch_ids = [enc.ids for enc in self._tok.encode_batch(texts)]
        if add_bos or add_eos:
            out = []
            for ids in batch_ids:
                if add_bos:
                    ids = [self.bos_id] + ids
                if add_eos:
                    ids = ids + [self.eos_id]
                out.append(ids)
            return out
        return batch_ids

    def decode(self, ids, skip_special=False):
        # CAMBIO 2026-08-22: .strip() (no .rstrip()) se comia la indentacion
        # inicial de la continuacion decodificada -- mismo bug que el de
        # generar() en benchmark_bcc_es.py (ver su comentario "CAMBIO
        # 2026-08-22"), pero aqui, en decode(), asi que ese fix no lo cubria:
        # HumanEval con Alfa (tokenizer nativo) seguia dando 0/80 por
        # IndentationError/SyntaxError mientras Qwen/Pythia (tokenizer HF,
        # sin este strip en su decode()) ya salian bien. rstrip() conserva
        # la indentacion inicial y solo recorta el sobrante final.
        txt = self._tok.decode([int(i) for i in ids], skip_special_tokens=skip_special)
        return txt.replace(self.user_token, "Usuario:").replace(self.assistant_token, "Sancho:").rstrip()

    def vocab_size(self):
        return self._tok.get_vocab_size()

    def get_stoi(self):
        return self._tok.get_vocab()

    def get_state(self):
        return {
            "type": "bpe_hf",
            "tokenizer_json": self._tok.to_str(),
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "user_token": self.user_token,
            "assistant_token": self.assistant_token,
            "fin_token": self.fin_token,
            "tool_token": self.tool_token,
            "tool_end_token": self.tool_end_token,
            "turno_token": self.turno_token,
            "story_token": self.story_token,
            "special_tokens": self.special_tokens,
        }

    def load_state(self, state):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_str(state["tokenizer_json"])
        self.pad_token = state.get("pad_token", "<PAD>")
        self.unk_token = state.get("unk_token", "<UNK>")
        self.bos_token = state.get("bos_token", "<BOS>")
        self.eos_token = state.get("eos_token", "<EOS>")
        self.user_token = state.get("user_token", "<USER>")
        self.assistant_token = state.get("assistant_token", "<ASSISTANT>")
        self.fin_token = state.get("fin_token", "<FIN>")
        self.tool_token = state.get("tool_token", "<tool>")
        self.tool_end_token = state.get("tool_end_token", "</tool>")
        self.turno_token = state.get("turno_token", "<turno>")
        self.story_token = state.get("story_token", "<|endofstory|>")
        self.special_tokens = state.get("special_tokens", [
            self.pad_token, self.unk_token, self.bos_token, self.eos_token,
            self.user_token, self.assistant_token, self.fin_token,
            self.tool_token, self.tool_end_token, self.turno_token,
            self.story_token])

    @property
    def pad_id(self): return self._tok.token_to_id(self.pad_token)
    @property
    def unk_id(self): return self._tok.token_to_id(self.unk_token)
    @property
    def fin_id(self): return self._tok.token_to_id(self.fin_token)
    @property
    def bos_id(self): return self._tok.token_to_id(self.bos_token)
    @property
    def eos_id(self): return self._tok.token_to_id(self.eos_token)
    @property
    def tool_id(self): return self._tok.token_to_id(self.tool_token)
    @property
    def tool_end_id(self): return self._tok.token_to_id(self.tool_end_token)
    @property
    def turno_id(self): return self._tok.token_to_id(self.turno_token)
    @property
    def story_id(self): return self._tok.token_to_id(self.story_token)

    def save_json(self, path):
        self._tok.save(path)

    def load_json(self, path):
        self._tok = Tokenizer.from_file(path)
