import glob
import os
import torch
# OJO: se importa de config_annealing, NO de config_pretrain. El punto de
# partida del SFT es el checkpoint ANNEALED (BLOCK_SIZE=8192), no el de
# pretrain crudo (BLOCK_SIZE=4096) -- si se importa de config_pretrain la
# arquitectura construida no coincide con el state_dict del checkpoint de
# origen. Ver config_annealing.py linea 80-85.
from alfa1.config_annealing import (
    N_EMBD, N_HEAD, N_LAYER, N_KV_HEAD, FFN_HIDDEN, TIE_WEIGHTS, ROPE_THETA,
    BLOCK_SIZE, BPE_VOCAB_SIZE, DROPOUT, DEVICE,
    CHECKPOINT_PATH as ANNEALING_CHECKPOINT_PATH,
)

# ============================================================
# Sancho/Alfa1 - SFT completo (full fine-tuning, sin LoRA)
# ============================================================
#
# La arquitectura se importa DIRECTAMENTE de config_pretrain.py (misma
# fuente de verdad) para que nunca pueda desincronizarse del checkpoint
# del que partimos. No la redefinas aqui.
#
# Corpus SFT: .txt YA FORMATEADOS con <USER>...<ASSISTANT>...<FIN>.
# Sin separador de documento aparte: <FIN> ya delimita cada ejemplo, asi
# que train_sft_bpe.py trocea por <FIN>. Mismo formato que sft_lora_alpha1.py,
# para no tener dos pipelines distintos.
# ============================================================

# --- Punto de partida (checkpoint de pretraining) y tokenizer (48k, congelado) ---
# 2026-09-07 (Exp.17): el ..._best.pt ya no existe (se limpio; queda copia del
# final en checkpoints_midtrain\ y en D:\). El final (iteracion=14976, ultimo
# paso del decay 1-sqrt a MIN_LR=1e-6, best_val_loss=1.1322) es equivalente al
# best a efectos practicos -> se usa como punto de partida del SFT.
SFT_INIT_FROM = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints_midtrain\alpha_1_annealing_05B.pt"
TOKENIZER_STATE_PATH = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\Generar_corpus\corpus_80Bt\bin_alfa1\tokenizer_state.json"

# --- Datos SFT: carpeta con los .txt ya formateados (<USER>...<ASSISTANT>...<FIN>) ---
# EXPERIMENTO 3: solo Alpaca (alpaca_es + alpaca_en, 62.091 pares), para
# comparar contra el Experimento 2 (solo Grok, 2.852 pares) con un corpus
# mucho mas grande y generico -- si Alpaca generaliza mejor en JSON/
# clasificacion/seguridad que el refuerzo manual de Grok (sospecha: Grok
# estaba memorizando plantillas muy parecidas al propio test, no
# generalizando), seria evidencia de que el problema de fondo era volumen
# de datos, no solo el contenido de las categorias. Generado por
# D:/Generar_SFT/unificar_corpus_solo_alpaca.py +
# D:/Generar_SFT/convertir_corpus_instrucciones_a_txt.py
#
# Corpus completo (Experimento 1, 157.451 pares) y solo-Grok (Experimento 2)
# quedan comentados por si hace falta volver a ellos:
# SFT_DATA_DIR = r"D:\corpus_instrucciones_alfa1_v2\sft_txt"
# SFT_DATA_DIR = r"D:\corpus_instrucciones_grok_only\sft_txt"
# SFT_DATA_DIR = r"D:\corpus_instrucciones_alpaca_only\sft_txt"  # Exp. 3
# Release v31 auditada: ReAct es el único bloque de agente; split por
# contenido, traducciones que conservan persona/tiempo y manifiesto reproducible.
# SFT_DATA_DIR = r"D:\alfa_sft_agent_v31\sft_txt"
# v32 (nombre reutilizado 2026-09-18, el intento anterior con este numero se
# descarto por regresion KNOW/SEARCH, ver memoria del proyecto): v31 + 116
# ejemplos nuevos auditados (identidad 32, opinion_preferencia 30,
# narrativa_libre 24, responder_directo refuerzo 30) cubriendo 3 huecos de
# KNOW al 0% encontrados en vivo (identidad, opinion, narrativa libre).
# SFT_DATA_DIR = r"D:\alfa_sft_agent_v32\sft_txt"
# v33 (2026-09-18): v32 + 81 ejemplos de abstencion "Observation fuera de
# tema", construidos con el buscar() REAL de produccion (no inventados),
# verificados uno a uno para no solapar con basura ya cubierta. Objetivo:
# subir el subtipo "fuera de tema" de 31 a ~110 sobre el total del bloque
# ReAct sin superar ~12% de abstencion total (leccion Hammer / ver memoria
# react-corpus-mejoras-literatura).
# SFT_DATA_DIR = r"D:\alfa_sft_agent_v33\sft_txt"
# v34 (2026-09-19): v33 con limpieza del bloque ReAct base (v31), a partir
# de una auditoria cruzada con GPT verificada a mano antes de aplicar (ver
# memoria del proyecto). 14 ejemplos BORRADOS (razonamiento no sustentado
# por la Observation, o contradiccion "ya tengo la informacion" -> "no
# tengo informacion"). 35 ejemplos con contaminacion de ingles en lang="es"
# TRADUCIDOS (contenido valido, solo el idioma estaba mal), incluyendo una
# correccion factual real de paso ("Los girasoles ciegos" es de Alberto
# Mendez, no Isabel Allende). No cambia el enrutado KNOW/SEARCH ni anade
# categorias -- riesgo bajo de regresion, es limpieza, no composicion nueva.
# v34 (2026-09-19, RECONSTRUIDO): el v34 anterior (limpieza GPT del bloque
# ReAct base) se borro y el numero se reutiliza para algo distinto: v33 -
# buscar_y_calcular (199, demasiado complejo para estos modelos, 25-28% de
# contradiccion razonamiento-vs-respuesta con Gemma, ver memoria
# completar-v33b-hallazgos) + buscar_factual REGENERADO end-to-end con datos
# reales (327 ejemplos): Alfa1 ejecuta la busqueda real (nunca inventada),
# Gemma-2-9B razona y responde SOLO a partir del fragmento real devuelto
# (nunca de su propio conocimiento), y una segunda pasada de auditoria +
# correccion manual (2 casos, ver memoria) cierra el lote. Pipeline:
# construir_release_sft_auditada.py --react-jsonl react_agente_es_train_v34.jsonl.
# Validacion: 0 fallos, 0 solapamiento train/val, traduccion y matematicas PASS.
# v35 (2026-09-19): CORRIGE el colapso de busqueda de v34. v34 quito
# buscar_y_calcular y dejo buscar[] en 42% del bloque ReAct -> el modelo casi
# dejo de buscar en modo agente (2/29 vs 17/29 de v30). v35 anade 200
# ejemplos buscar_factual NUEVOS (50 historia/filosofia/literatura/ingenieria)
# con Observation REAL de Wikipedia + razonamiento/respuesta de Gemma-2-9B
# (11 fallos de extraccion corregidos a mano en auditoria), restaurando
# buscar[] al 51.3% (identico a v33). Ver memoria v34-resultado-entreno-y-replay.
# v36 (2026-09-19): quita las CONTRADICCIONES de etiqueta que quedaban en
# v35. El corpus tenia la MISMA forma de pregunta con etiqueta opuesta:
# "quien escribio/es autor de X" 17 en responder_directo (NO busca) vs 79 en
# buscar_factual (SI busca); "cual es el/la mas ... del mundo" 10 vs 0 (por
# eso "rio mas largo" respondia directo y alucinaba "Congo"); y 3 preguntas
# LITERALMENTE identicas con etiqueta opuesta. Distinguirlas exige
# autoconocimiento calibrado que un 520M no tiene -> el modelo lanzaba una
# moneda (la misma pregunta de Google buscaba con "?" y no sin "?").
# FIX: se eliminan de responder_directo esas 41 trazas; la regla pasa a ser
# aprendible por FORMA superficial (autoria/superlativo -> siempre busca) y
# responder_directo se queda solo con lo trivial (capitales, simbolos
# quimicos, meses, organos). buscar[] 53.0%, 0 contradicciones literales.
# v37 (2026-09-20): arregla un DEFECTO GRAVE que introduje en el banco de
# v35: las 200 queries que escribi llevaban LA RESPUESTA dentro
#   "¿Quien escribio Cien anios de soledad?" -> buscar[Cien anios de soledad Garcia Marquez]
# Eso ensena la regla "query = tema + respuesta"; al inferir, si no sabe la
# respuesta, se la INVENTA para rellenar el hueco -> observado en v36:
# `buscar[La Regenta Miguel de Cervantes]`. Es imposible poner en la consulta
# el dato que vas a averiguar CON esa consulta.
# FIX: se SUSTITUYEN esos 200 (ids 2000-2199) por 220 nuevos con query
# limpia (solo terminos de la pregunta, verificado automaticamente: 0
# filtraciones) + 20 de superlativo-del-mundo, hueco que quedo abierto en
# v36. buscar[] 53.7%, 0 contradicciones literales.
# EFECTO COLATERAL INHERENTE: con queries honestas el indice recupera peor,
# asi que la abstencion dentro de buscar_factual sube de 42.1% a 50.1%. No es
# una eleccion de diseno: es el recall real del indice quedando al
# descubierto (en v35 la busqueda iba "guiada" por la respuesta en la query).
# v38 (2026-09-20): dos correcciones sobre el hallazgo de plantilla
# memorizada en buscar_factual (ver memoria hallazgo-plantilla-abstencion-
# memorizada): 274/547 (50.1%) de las abstenciones en v37 eran la MISMA
# frase, caracter por caracter -- causaba que el Thought a veces extrajera
# bien el dato (Tesla, Google) y el Final Answer abstuviera igual. FIX:
# se regeneraron esas 274 pidiendo a Gemma que abstenga con SUS PROPIAS
# palabras, mencionando que falta especificamente (mismo patron que ya
# funciono para rechazos_seguridad). Resultado: 273/274 frases unicas
# (99.6%, antes 0%). Ademas se quito buscar_y_calcular_abstencion (120,
# huerfana desde que se retiro buscar_y_calcular en v34 -- sin ningun caso
# de EXITO de esa familia, solo ensenaba a abstener siempre). v38: 1121
# ejemplos ReAct, 0 fallos de validacion, 0 solapamiento train/val.
# v39 (2026-09-21): CAMBIO DE FORMATO del bloque de agente, no de contenido.
# El corpus es EXACTAMENTE el mismo que v38 (92.341 ejemplos, mismos
# recuentos train/val exactos) -- lo unico que cambia es como se representa
# el bucle de agente, de texto libre ReAct al formato estructurado estilo
# Qwen/Llama:
#   v38: <USER>{preambulo}Pregunta: X<ASSISTANT>Thought: ...\nAction: buscar[q]
#        <USER>Observation: {frag}<ASSISTANT>Thought: ...\nFinal Answer: R<FIN>
#   v39: <USER>{persona + tools como JSON Schema}Pregunta: X
#        <ASSISTANT><tool>{"name":"buscar","arguments":{"consulta":"q"}}</tool>
#        <turno>{frag}<ASSISTANT>{razonamiento en prosa}\n\nR<FIN>
# Motivos (ver conversacion del 2026-09-20/21):
#  (a) La llamada pasa a ser JSON validable -> se puede forzar con gramatica
#      restringida en inferencia (literatura: dependencia dura para
#      tool-calling fiable en modelos pequenos). "Action: buscar[...]" en
#      texto libre no se puede restringir.
#  (b) El resultado de la herramienta deja de disfrazarse de <USER>
#      Observation: y usa <turno> (token 9, reservado y sin uso hasta hoy).
#      El modelo ya no confunde un resultado con una pregunta del usuario.
#  (c) Se elimina la plantilla "Thought:" ANTES de la Action, que es donde
#      vivian los DOS colapsos de plantilla documentados (547/942 vacios y
#      el resto 3 plantillas fijas, a menudo en ingles; ademas las 212
#      sustituciones de normalizar_react.py caen TODAS ahi). El razonamiento
#      previo a la respuesta final SI se conserva, en prosa y sin etiqueta:
#      son 726 ejemplos (65%) con ~289 caracteres de media, y es justo donde
#      se razona sobre el fragmento recuperado.
# NOTA: los tokens <tool>/</tool>/<turno> (ids 7/8/9) YA estaban reservados
# en el tokenizer congelado, asi que NO hace falta reentrenarlo y el val
# loss sigue siendo comparable con toda la serie v31-v38.
# Construido con: construir_release_sft_auditada.py --react-v39
# Validacion: 0 fallos, 942 tool_calls verificadas, 0 solapamiento train/val.
# v40 (2026-09-21): FRONTERA DE COMPROMISO. Corrige una regresion que
# introdujo v39 y que se midio con 5 corridas por caso:
#     caso                          v38    v39
#     Calderon, fragmento largo     2/5    4/5
#     Calderon, fragmento minimo    5/5    3/5
#     Google 1998, fragmento corto  5/5    0/5   <- no es varianza
# v39 fusiono razonamiento y respuesta en un unico tramo <ASSISTANT>. El
# marcador "Final Answer:" de v38 no era solo plantilla: era una frontera que
# obligaba a cerrar y comprometerse. Al quitarla, el bloque a generar paso de
# 85 caracteres de media (solo respuesta, v38) a 444 (fusionado, v39) -- 5x
# mas tokens para derivar hacia la abstencion antes de comprometerse. Sintoma:
#   "El fragmento menciona que Google fue fundada en 1998... Sin embargo, no
#    dice si esa fue la fecha... Por lo tanto, no puedo extraer el dato."
# v40 separa razonamiento y respuesta en DOS tramos <ASSISTANT> seguidos, sin
# reintroducir "Thought:" ni ninguna plantilla de texto:
#   <turno>{frag}<ASSISTANT>{razonamiento}<ASSISTANT>{respuesta}<FIN>
# El tramo de respuesta vuelve a 132 caracteres de media. Corpus IDENTICO a
# v38/v39 (92.341 ejemplos, mismos recuentos): solo cambia la frontera, asi
# que la comparacion v38/v39/v40 queda aislada.
#
# v41 (2026-09-22): el bloque del agente corregido A MANO ejemplo a ejemplo
# (746 de 1121: errores factuales que se ensenaban como ciertos, queries
# corruptas, falsos negativos y tres plantillas de Gemma). El master/ es bit a
# bit IDENTICO al de v40 (mismos sha256) y el split da los mismos recuentos,
# asi que la comparacion v40 vs v41 aisla el efecto del corpus del agente.
# v42a (2026-09-23). Se sale del bloque de agente y se ataca el SFT base, que
# es donde los benchmarks externos senalan el problema: IFEval-es 0,139
# prompt-strict, con 0-5 % en TODO lo que exige producir estructura (vinetas,
# parrafos, JSON, mayusculas/minusculas, terminar con frase exacta).
# Tres cambios y nada mas, para que el IFEval sea atribuible:
#   + OASST2-es humano, 3.973 (mediana 877 car vs 214 del corpus, 22,8 % con
#     markdown vs 1,9 %, 21,3 % multiturno vs 11,5 %)
#   + smol-constraints EN, 8.983 (restricciones verificables; en ingles a
#     proposito, para medir si el formato transfiere de idioma)
#   - higiene: 242 ejemplos de alpaca con caracteres CJK incrustados
# El bloque de agente es el MISMO fichero de v41, byte a byte.
SFT_DATA_DIR = r"D:\alfa_sft_agent_v43\sft_txt"
# SFT_DATA_DIR = r"D:\alfa_sft_agent_v41\sft_txt"
# SFT_DATA_DIR = r"D:\alfa_sft_agent_v40\sft_txt"
# SFT_DATA_DIR = r"D:\alfa_sft_agent_v39\sft_txt"
# SFT_DATA_DIR = r"D:\alfa_sft_agent_v38\sft_txt"
# SFT_DATA_DIR = r"D:\alfa_sft_agent_v37\sft_txt"
SFT_DATA_FILES = sorted(glob.glob(os.path.join(SFT_DATA_DIR, "*.txt")))

# --- Tokens especiales (deben existir en el vocab del tokenizer 48k) ---
USER_TAG = "<USER>"
ASSISTANT_TAG = "<ASSISTANT>"
TURN_TAG = "<turno>"
FIN_TAG = "<FIN>"
PAD_TAG = "<PAD>"
# v39 (2026-09-21): formato de agente estructurado estilo Qwen/Llama.
# TOOL_TAG/TOOL_END_TAG envuelven la llamada JSON del propio asistente
# (se entrenan como generacion normal, van DENTRO de un tramo <ASSISTANT>).
# TURN_TAG marca el turno de ENTRADA con el resultado de la herramienta
# (sustituye a "disfrazarlo" de <USER>Observation: ..., que confundia el
# resultado real con una pregunta de usuario). train_sft_bpe.py enmascara
# la loss desde <turno> hasta el siguiente <ASSISTANT>, igual que ya hacia
# con <USER>.
TOOL_TAG = "<tool>"
TOOL_END_TAG = "</tool>"

# Si True, la loss solo se calcula sobre <ASSISTANT>...<FIN> (se enmascara
# el prompt <USER>...). NO usa el ignore_index interno de model.py (no
# confirmado cual usa) -- train_sft_bpe.py lo recalcula a mano sobre los
# logits. Ver el comentario al principio de ese fichero.
MASK_PROMPT_LOSS = True

# --- Split ---
TRAIN_RATIO = 0.92

# --- Entrenamiento ---
# OJO: BATCH_SIZE=8 estaba calibrado para BLOCK_SIZE=4096 (config_pretrain,
# de cuando este fichero importaba mal la arquitectura). Con el fix a
# config_annealing, BLOCK_SIZE=8192 real -- el doble de contexto duplica
# aprox. la memoria por secuencia, y con full fine-tuning (gradientes +
# estados de AdamW, sin LoRA) esto provoco un OOM real al primer batch:
# "CUDA out of memory... 41.94 GiB is allocated" en una GPU de 31.84 GiB
# (log_sft_pipeline.txt, 2026-09-04). Bajado a BATCH_SIZE=2 y subido
# GRAD_ACCUM_STEPS a 24 para mantener el mismo batch efectivo (48
# secuencias/update). Vigila nvidia-smi al relanzar; si sigue habiendo OOM,
# bajar mas.
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 24

# Reactivado: el OOM lo causaba BATCH_SIZE=8 (memoria), no torch.compile.
# Confirmado que BATCH_SIZE=2 evita el OOM por si solo (corrida sin compile
# llego a iter 350/6034 sin problemas, a ~6.600 tok/s -- lejos de los
# ~33.000 tok/s del annealing). Se reactiva compile para recuperar
# velocidad; las formas dinamicas (longitud variable por batch) pueden
# hacer que tarde mas en compilar los primeros pasos, pero no deberia
# volver a dar OOM ya que la causa raiz era el batch grande, no esto.
#
# EXCEPCION 2026-09-05 (Experimento 2, corpus solo Grok, 106 iteraciones):
# crash "CUDA driver error: out of memory" dentro de un kernel Triton al
# cambiar de forma train->eval (torch.compile recompila para cada longitud
# distinta de secuencia, y ese recompilado transitorio fragmenta VRAM justo
# en el borde entre formas). Con tan pocas iteraciones el coste de no
# compilar era insignificante, asi que se desactivo para esa corrida.
#
# Experimento 3 (solo Alpaca, 62.091 pares): con un corpus 22x mas grande
# el numero de iteraciones sube mucho y SI compensa compilar -- se
# reactiva. El riesgo de OOM al cambiar de forma sigue existiendo (no se ha
# arreglado la causa raiz), asi que si vuelve a saltar el mismo error habra
# que desactivarlo de nuevo o investigar por que en serio.
#
# EXPERIMENTO 15 (2026-09-06): se probo a DESACTIVARLO por un falso
# diagnostico -- las primeras iteraciones iban a ~250-270 tok/s y se
# interpreto como un fallo de recompilacion de torch.compile. En realidad
# es solo el arranque en frio: el log del Exp.14_v2 arranca IGUAL (iter 1 a
# 242 s/it, tok_s=259) y rampa a 4.135 tok/s en iter 300, 10.799 en iter
# 600 y ~12.000 estable a partir de iter 900 (warmup del LR + autotuning de
# cuDNN + crecimiento del asignador + compilacion del grafo). Sin compile
# la rampa es igual pero el techo es algo mas bajo (~1.5-2x, ver arriba).
# Para la corrida definitiva del Exp.15 se vuelve a compile=True (config
# del Exp.14, conocida y buena). [Nota: la corrida lanzada el 2026-09-06
# 20:xx corrio con compile=False por este episodio; si su resultado es
# bueno, sirve igual.]
COMPILE_MODEL = True

EPOCHS = 2                 # EXPERIMENTO 20 (2026-09-08): se probo EPOCHS=3 con
                            # el corpus de v16 exacto. RESULTADO: best val loss
                            # 1.1831 (v16 1.1883, mejora 0.005 = ruido) alcanzado
                            # al FINAL de la 2a epoca; la 3a sube el val loss a
                            # ~1.20 (train loss cae a 0.7 = memorizacion) y el
                            # benchmark queda en v16 +-ruido. La 3a epoca NO
                            # aporta. Revertido a 2. Ver
                            # Benchmark_SFT/resultados/scoring_v16_vs_v20_3epocas.md

EVAL_INTERVAL = 300
EVAL_ITERS = 50
GRAD_CLIP = 1.0

# --- LR: mucho mas bajo que en pretraining, warmup corto ---
# EXPERIMENTO 7/8 (2026-09-05): LR a la mitad (1e-5) para probar si reducia
# la inestabilidad entre runs -- DESCARTADO. Val loss se estanco en ~1.29
# incluso con PATIENCE=10 (Exp.8, sin early stop, 2740 iters completas),
# claramente peor que el 1.19 de LR=2e-5. No era falta de paciencia, el LR
# bajo converge a un techo peor. Se vuelve al LR=2e-5 original.
#
# EXPERIMENTO 9 (2026-09-05): corpus "alfa_sft_agent" (formato Hermes-style,
# con email/procedimiento/agente_tool_use reforzados tras confirmar que
# nunca se resolvieron en 6 fases anteriores) + LR=2e-5 (el bueno, no el
# 1e-5 descartado) + shuffle estable + PATIENCE=5 normal. Primera vez que
# se combina el formato correcto para HF con el LR que si funciona.
LEARNING_RATE = 2e-5
MIN_LR = 2e-6
WARMUP_ITERS = 100
WEIGHT_DECAY = 0.01         # menor que en pretraining (0.1); ya viene regularizado
BETA1 = 0.9
BETA2 = 0.95

# --- Early stopping: aqui SI importa (a diferencia del pretraining, con
# poco volumen de datos el overfitting es real -- ver conversacion) ---
# Vuelto a PATIENCE=5 (normal) para el Exp.9, ya con el LR bueno (2e-5).
EARLY_STOPPING = True
PATIENCE = 5
MIN_DELTA = 0.001

# --- Checkpoints ---
# Nombre distinto de "alpha_1_sft_full.pt" (Exp. 1) y "alpha_1_sft_grok.pt"
# (Exp. 2) para no sobreescribir ninguno -- este es el Experimento 3 (solo Alpaca).
# EXPERIMENTO 5 (2026-09-05): mismo corpus que Exp.4 (Alpaca+ con degradacion,
# 4.95/10) pero con agente_tool_use y busqueda_usar ampliados (5->22 y 8->16
# ejemplos/idioma) y sobremuestreados (x4 y x3) tras medir 0% y 17-50% en esas
# categorias -- fallo por pura escasez, no por dificultad de la tarea.
# RESULTADO: 4.57/10 (bajo desde 4.95), busqueda_usar_es mejoro mucho
# (1/6->6/6) pero hubo regresiones fuertes en seguridad_es (8->2) y
# agente_clasificador (9->4, 7->2) -- probablemente por reordenamiento de
# todo el flujo de entreno al cambiar el tamano del corpus, no por los datos
# nuevos en si. No se adopta el checkpoint v5 para uso real.
#
# EXPERIMENTO 6 (2026-09-05): mismo corpus base + traduccion bidireccional
# ES<->EN (212, agujero: general_traduccion_es 10%), email (140, agujero:
# general_email 0% en las 4 fases previas) y procedimiento (120 tras
# sobremuestreo x4, agujero: general_procedimiento 0% en las 4 fases previas,
# para comprobar si es falta de datos o techo real del modelo).
#
# EXPERIMENTO 7 (2026-09-05): mismo corpus que Exp.6 pero con el formato de
# tool-calling cambiado a Hermes-style (<tool_call>{"name":...,"arguments":
# ...}</tool_call>, ver runtime_agente.py) de cara a compatibilidad con
# publicacion futura en Hugging Face (vLLM/llama.cpp ya traen parser para
# ese formato de facto). Ademas usa shuffle estable por hash (no
# random.shuffle, ver train_sft_bpe.py) y LR=1e-5 en vez de 2e-5, para
# aislar si la inestabilidad entre runs viene del LR o del reordenamiento.
# RESULTADO: 3.76/10, el PEOR de los 4 ultimos experimentos. Early stopping
# corto en iter 2100/2740 con val loss 1.2923 (peor que 1.1922 del Exp.6).
# La volatilidad NO bajo -- categorias solidas como control_identidad_en
# (17->8) y general_resumen_contexto_es (11->0) se desplomaron. Diagnostico:
# no fue una prueba justa del LR, el modelo se quedo infra-entrenado porque
# el early stopping (PATIENCE=5) corto antes de que un LR mas bajo pudiera
# converger. Se repite como Exp.8 con PATIENCE=10.
#
# EXPERIMENTO 8 (2026-09-05): identico al Exp.7 (mismo corpus, formato
# Hermes, shuffle estable, LR=1e-5) pero con PATIENCE=10 en vez de 5, para
# dar tiempo real de convergencia antes de asumir que el LR bajo no mejora.
# DATASET FINAL: renombrado a "alfa_sft_agent" (2026-09-05) -- ver
# D:\alfa_sft_agent\README.md para fuentes y evaluacion completa.
#
# EXPERIMENTO 9 (2026-09-05): mismo corpus + LR=2e-5 (el bueno) + shuffle
# estable + PATIENCE=5. RESULTADO: 5.10/10, practicamente igual al Exp.6
# (5.08-5.24/10 segun redondeo) -- ninguna mejora neta pese a arreglar LR y
# formato Hermes a la vez. El checkpoint de referencia real seguia siendo
# alpha_1_sft_alpaca_plus_v6_best.pt hasta este fix.
#
# EXPERIMENTO 10 (2026-09-05): IDENTICO al Exp.9 en corpus/LR/shuffle/
# patience -- el UNICO cambio es el fix de train_sft_bpe.py:119
# (tokenizer.encode(doc) -> tokenizer.encode(doc, normalize_chat=False)).
# Auditoria del pipeline GRPO detecto que normalize_chat=True hacia
# raw_line.strip() en CADA linea del documento, destruyendo la indentacion
# de las 5.500 respuestas de codigo del corpus (7.6%) en TODOS los
# experimentos anteriores (1-9). Confirmado con el checkpoint v9: 0/40
# respuestas de codigo generadas compilaban ("expected an indented block").
# Verificado que el fix preserva indentacion en ejemplos reales del corpus
# y que <USER>/<ASSISTANT>/<FIN> se siguen detectando como tokens unicos
# (el enmascarado de loss no se ve afectado). Si dominio_programacion sube
# de forma notable en el benchmark, confirma que el bug era la causa raiz
# del estancamiento en 6-8/16 durante 9 fases; si no sube, descarta esta
# hipotesis y el problema es de otra naturaleza (capacidad del modelo,
# volumen de datos de codigo, etc).
# RESULTADO Exp.10: 1.2510 val loss (mejor hasta la fecha), confirmado que
# el bug de indentacion era real. Checkpoint v10 fue ademas la base sobre la
# que se corrio GRPO-Exp1 (no mejoro, ver
# Benchmark_SFT/resultados/respuestas_grpo_exp1_SINTRUNCAR.json).
#
# EXPERIMENTO 11 (2026-09-06): mismo corpus base que Exp.10 + refuerzo de
# las 9 causas raiz de dominios flojos detectadas en la auditoria del
# benchmark v10 SIN TRUNCAR (ver OVERSAMPLE en unificar_corpus_alpaca_plus.py
# y D:/Generar_SFT/refuerzo_dominios/). Ademas se corrigio un bug real en
# convertir_corpus_instrucciones_a_txt.py: solo tomaba el PRIMER turno de
# messages, descartando el resto -- todo ejemplo multiturno de TODOS los
# experimentos anteriores (Exp.4-10) entrenaba solo con el turno trivial de
# apertura, nunca con el turno de refuerzo real. Corregido antes de este
# experimento: 631 documentos multiturno ahora llevan ambos turnos.
# RESULTADO Exp.11: val loss 1.1358 (mejor hasta la fecha, vs 1.2510 de
# v10). En benchmark, +4.5pp global (57.8% -> 62.3%), pero mixto: mejoras
# claras en control_identidad (40->90 ES), traduccion_en (30->60),
# social_en/tool-call espurio y multiturno_en (20->40); sin cambio en
# general_json (17%) y general_procedimiento (0%); y regresiones en
# clasificacion_en, alucinacion_en, restricciones_en, titulador_en y
# opinion_en.
#
# EXPERIMENTO 12 (2026-09-06): dos correcciones dirigidas a fallos MEDIDOS,
# no a hipotesis:
#
#  (a) general_json seguia en 17% en v10 Y v11. Al separar formato de
#      contenido (nuevo Benchmark_SFT/evaluar_formato.py) se vio que el
#      formato era 100% en AMBOS: el modelo siempre emitia JSON valido con
#      las claves correctas, y el fallo era solo de VALORES -- 12/12 items
#      respondieron "momento": "hoy", enum colapsado. Causa: el corpus
#      json_enum del Exp.11 enseñaba JSON de UN campo, mientras la tarea
#      real es de dos campos con null y extraccion de entidad. Se retira ese
#      corpus y lo sustituye generar_json_extraccion.py (600 items,
#      multi-campo, null en 261, enum balanceado por construccion, claves en
#      español con pregunta en ingles como en el benchmark).
#
#  (b) REGRESION causada por el propio refuerzo del Exp.11: el corpus de
#      alucinacion enseño a responder "no tengo acceso a datos en tiempo
#      real", y el modelo empezo a rechazar en vez de USAR la herramienta de
#      busqueda disponible (busqueda_usar_en, cumplimiento de formato:
#      100% -> 33%; en un item llego a inventarse un resultado de F1 con
#      fecha y circuito). Ningun ejemplo enseñaba a distinguir los casos. Lo
#      arregla generar_contraste_busqueda_alucinacion.py con un contraste de
#      tres vias: hay herramienta + consultable -> llamarla; no hay
#      herramienta -> admitir; hay herramienta + imposible de consultar
#      (futuro, ficticio, pensamientos) -> admitir SIN llamarla (este tercer
#      caso evita pasarse de frenada y romper busqueda_no_usar).
# RESULTADO Exp.12: val loss 1.1376. Benchmark contenido 5.81/10 (vs 5.70
# de v10, 5.47 de v11). JSON arreglado del todo (3-4/12 -> 12/12 ambos
# idiomas), pero el contraste busqueda/alucinacion salio MAL: solo 6 frases
# de rechazo unicas en el caso B con oversample x3 -> el modelo las memorizo
# y las emitia fuera de lugar (15 items contaminados, 12 fuera de busqueda;
# busqueda_usar_en 100%->33% en contenido). Ademas destapo (via el fix de
# truncado) un agujero de seguridad real: explosivos y malware obedecidos en
# ambos idiomas, en las tres versiones -- solo se rechazan los temas con
# ejemplos en el corpus de Grok. El corpus de seguridad lo prepara el
# usuario aparte (decision del 2026-09-06), NO se genera aqui.
#
# EXPERIMENTO 13 (2026-09-06): mismo corpus que Exp.12 salvo el contraste
# busqueda/alucinacion, sustituido por su v2 (generar_contraste_busqueda_
# alucinacion.py): respuestas compuestas por partes (111 frases unicas en el
# caso B en vez de 6, 149 en el C), reparto 2:1:1 a favor de "usa la
# herramienta", y oversample bajado (contraste a x1, busqueda_usar de x3 a
# x1) para no reintroducir la saturacion que causo la memorizacion. Se
# corrigio ademas un bug del generador: CASOS[i % 3] sobre una lista de 4
# casos dejaba el caso C sin generar. Objetivo aislado: comprobar que la
# regresion de busqueda queda resuelta sin tocar seguridad.
# RESULTADO Exp.13: val loss 1.1551. El fix del contraste busqueda/
# alucinacion (v2, mas variado) no se pudo confirmar limpiamente por
# benchmark (formato de busqueda_usar_es empeoro 67%->0% en esa medicion
# puntual), pero SI se confirmo en conversacion real posterior: v13 ya
# distingue bien cuando debe llamar a una herramienta y cuando debe admitir
# incertidumbre en los casos entrenados (aunque no generaliza a frases muy
# distintas como "busca informacion sobre X", ver mas abajo).
#
# EXPERIMENTO 14 (2026-09-06): PRIMERA VEZ que el refuerzo viene de
# CONVERSACION REAL (chat_web_app.py + logs_chat_web/), no del benchmark de
# 316 items. Se detectaron cuatro fallos nuevos que el benchmark nunca
# pudo medir por probar solo turnos aislados o pares cortos:
#
#  (a) Con solo 2 turnos de historial y contexto corto (~59 tokens), una
#      pregunta de la MISMA forma que la anterior (quien/cuando/que es)
#      sobre un tema DISTINTO hacia que el modelo repitiera la respuesta
#      vieja en vez de procesar la nueva ("quien invento la maquina de
#      vapor" -> "cuando empezo la 1a guerra mundial" devolvio la respuesta
#      sobre la maquina de vapor). No es el problema de contexto largo
#      (eso ya se mitigo limitando MAX_HISTORY_TURNS en chat_web_app.py) --
#      es condicionamiento debil en el turno nuevo incluso a corto plazo.
#  (b) Contaminacion de identidad: tras mencionar un nombre propio en el
#      contexto, "¿Cual es tu nombre?" respondio con ESE nombre en vez de
#      la identidad del asistente.
#  (c) "Anclaje" a contenido alucinado propio: una instruccion nueva y
#      ajena (traducir una frase) se resolvia sobre el tema anterior en vez
#      de sobre si misma.
#  (d) "Titulo para esta noticia" y "Resume la noticia" sobre el MISMO
#      texto devolvieron la respuesta IDENTICA -- el modelo no distingue
#      esas instrucciones como tareas con formas de salida distintas.
#
# Corpus nuevos: generar_cambio_tema.py (~1000 items programaticos, sin
# LLM, respuesta correcta conocida por construccion: cambio_tema,
# identidad_no_confundir, anti_anclaje) y generar_titulo_resumen_
# traduccion.py (1200 items via vLLM/Gemma-2-9B, trios del mismo parrafo
# con titulo/resumen/traduccion validados programaticamente como realmente
# distintos entre si).
# RESULTADO Exp.14: mejor checkpoint hasta la fecha -- 385/624 (61.7%) en el
# benchmark de contenido, con mejoras reales confirmadas en identidad,
# traduccion y titulo. Val loss 1.2132 (con la metrica de validacion YA
# CORREGIDA: estimate_loss() cubre ahora el split de validacion COMPLETO,
# no una muestra de 100 ejemplos -- ver train_sft_bpe.py:279-309, fix del
# 2026-09-06; por eso 1.2132 no es comparable con los ~1.13-1.15 de
# Exp.11-13, que se midieron sobre la muestra ruidosa). Quedaban sin
# arreglar el colapso titulo->resumen en secuencia y el manejo pobre de
# feedback negativo.
#
# EXPERIMENTO 15 (2026-09-06/07): cuatro refuerzos sobre el corpus del
# Exp.14 (unificar_corpus_alpaca_plus.py, TOTAL 89.328 ejemplos):
#
#  (a) titulo_resumen_secuencial (798 items via vLLM/Gemma-2-9B, x2):
#      arregla un colapso CONFIRMADO con evidencia real en v14 -- pedir un
#      titulo y, en el turno siguiente, un resumen del MISMO texto devolvia
#      la misma respuesta corta. El corpus del Exp.14
#      (titulo_resumen_traduccion) generaba los tres tipos como turnos
#      INDEPENDIENTES; este genera la SECUENCIA real de 2 turnos sobre el
#      mismo parrafo, en ambos ordenes.
#  (b) multiturno_extendido (500 items programaticos, x3): pedido explicito
#      del usuario -- extender multiturno mas alla de los 2 turnos que
#      tenia TODO el material anterior. Cadenas de 3-4 turnos con 1-2
#      turnos distractores entre el dato y la pregunta que lo recupera.
#  (c) rechazos_seguridad (64 items, x2-x3): agujero de seguridad REAL
#      confirmado en 3 benchmarks (Exp.12-14) -- el modelo obedecia
#      peticiones de explosivos y malware salvo en los temas con ejemplo en
#      el corpus de Grok. El usuario decidio el 2026-09-06 incluirlo ya en
#      esta ronda (antes lo iba a preparar aparte). Rechazo firme +
#      contraste educativo (explica el concepto sin instrucciones
#      operativas). Publicar el modelo solo como investigacion.
#  (d) feedback_negativo (400 items programaticos, x3): fallo real y
#      repetido en v13 y v14 (logs_chat_web/) -- ante "no me gusta ese
#      chiste" el modelo REDOBLABA justificando la respuesta anterior en
#      vez de reconocer la critica y ofrecer corregir el rumbo. 2 turnos,
#      respuesta compuesta por partes (reconoce + ajuste opcional +
#      ofrece), un tercio con pista de correccion explicita.
# RESULTADO Exp.15: val loss 1.1760 (mejor que 1.2132 de v14). Los 4
# fallos objetivo ARREGLADOS en el replay del log real. PERO el benchmark
# de 316 items (juez: Claude, misma pasada v14 y v15, ver
# Benchmark_SFT/resultados/scoring_v15_vs_v14.md) da 61.1% para v15 frente
# a 66.2% de v14: seguridad +8 (v14 daba instrucciones de explosivos y
# malware, v15 rechaza) pero regresiones reales en control_identidad
# (-4/-4: "desarrollado por OpenAI", inventa familia/residencia),
# control_alucinacion_es (-6: inventa poblacion y telefonos donde v14
# admitia no saber) y general_social_en (-4: el patron de feedback
# negativo se cuela en turnos que no son critica). v15 NO se adopta.
#
# EXPERIMENTO 16 (2026-09-07): mismo corpus que Exp.15 con dos ajustes
# dirigidos a esas regresiones (ver OVERSAMPLE en
# unificar_corpus_alpaca_plus.py):
#  (a) feedback_negativo BAJADO de x3 a x1 (1200 -> 400 items) y añadido
#      feedback_neutro (140 items de contraste: 3er turno que NO es
#      critica -> respuesta normal, sin disculpa) a x2. Corta la
#      sobre-generalizacion de "reconozco + ¿que quieres cambiar?".
#  (b) identidad de Grok (116 items) y safety (93) sobremuestreados por
#      primera vez a x4 / x3 -- con las ~4700 nuevas del Exp.15 su peso
#      relativo habia caido y el modelo volvio a confundir su identidad.
# Todo lo demas identico al Exp.15 (seguridad, multiturno, titulo/resumen
# secuencial sin cambios). TOTAL 89.342 ejemplos.
# RESULTADO Exp.16: val loss 1.1883. Benchmark (juez: Claude, misma pasada
# v14/v15/v16, ver Benchmark_SFT/resultados/scoring_v14_v15_v16.md) ~67%,
# igual o mejor que v14 (66.2%) y muy por encima de v15 (61.1%).
# control_seguridad 20/20 (jailbreak incluido), control_identidad 33/40
# (recuperado, sin "OpenAI" ni familia inventada). v16 ADOPTADO como
# referencia (movido a D:\alpha_1_05B_it_v16.pt) y exportado a GGUF.
#
# EXPERIMENTO 17 (2026-09-07): dos parches de datos pequenos y dirigidos
# sobre el corpus del Exp.16 (ver DIFF de unificar_corpus_alpaca_plus.py):
#  (a) saludo_breve (90 items programaticos, x3 -> 270): agujero TOTAL
#      detectado probando v16 en Ollama -- CERO ejemplos de "Hola"/"Buenas"
#      a secas como turno unico en las 89.342 muestras del Exp.16. El modelo
#      caia en "¡Que interesante! ¿Y tu?" (muletilla de Grok). Ver
#      generar_saludos.py.
#  (b) identidad de Grok x4 -> x5 (+116 items): v16 aun tuvo fugas
#      ("vivo en Madrid", edad humana) -> control_identidad 33/40.
# NADA MAS cambia respecto al Exp.16 (alucinacion y titulo/resumen quedan
# para una ronda posterior: necesitan mas que datos). TOTAL 89.728.
#
# RESULTADO Exp.17: val loss 1.1882 (empate con v16). Benchmark (juez: Claude,
# misma pasada v16/v17, ver Benchmark_SFT/resultados/scoring_v16_vs_v17.md):
# REGRESION ~-3 pts (~62% -> ~59%). Arregla "Hola" a secas (verificado) pero
# el oversample de identidad NO cerro su diana y ademas regresiono
# (control_identidad 35/40 -> 31/40, explicacion con mas invencion, social
# contaminado por el saludo). control_seguridad se mantuvo 20/20. v17 DESCARTADO.
#
# EXPERIMENTO 18 (2026-09-07): "v18 = v16, sin oversampling nuevo". identidad
# x4->x1, saludos x1/26, +26 reconoce_error x1. RESULTADO: val loss 1.1970,
# benchmark ~55-57%, control_identidad SE CAE a 21/40 (identidad_en 20->8/20:
# "born in Tokyo 1963", "I live in the United States", "my parents are both
# doctors"). El x4 de identidad es load-bearing. reconoce_error a x1 no cuaja
# (26/89000). Seguridad 20/20. v18 DESCARTADO. Ver scoring_v16_vs_v18.md.
#
# EXPERIMENTO 19 (2026-09-08): "v16 + saludos a x2". identidad se queda a x4
# (probado: subirla o bajarla empeora). Unico cambio vs Exp.16: saludo a
# secas a x2 con set mas variado -- 120 items, 46 formas ("hola"/"holaa"/
# "hola que tal"/"hi"/"hello"/"buenas"/"hey"/"buenos dias"/"como estas"...),
# 84 respuestas distintas. Punto medio entre v17 (x3/270 -> contamino
# general_social) y v18 (x1/26 -> fuga persistente a "Gracias"). Sin
# reconoce_error. Ver generar_saludos.py.
# RESULTADO Exp.19: val loss 1.1897. identidad RECUPERADA a 35/40 (x4 confirma
# como optimo), pero general_social se CONTAMINA 17->10/20 (peor que v17). El
# refuerzo de saludos contamina general_social a cualquier x que tape el bare
# "Hola". v19 DESCARTADO. Ver scoring_v16_vs_v19.md.
#
# EXPERIMENTO 20 (2026-09-08): EPOCHS 2->3 sobre corpus de v16. La 3a epoca
# no aporta (val loss 1.1831 en frontera de la 2a, luego sobre-ajusta).
# Revertido a EPOCHS=2. Ver scoring_v16_vs_v20_3epocas.md.
#
# EXPERIMENTO 21 (2026-09-08): "Opcion A" tras 2 rondas de DPO fallidas para
# identidad. DIAGNOSTICO: no era techo de 520M, era COBERTURA. Las sondas
# "¿donde vives?/tienes familia?/que edad/donde naciste/que hiciste el finde"
# solo existen en INGLES en el corpus (~5 frases memorizadas); en ESPANOL:
# CERO. Por eso v16 acierta id_en07 y falla id07 ("Vivo en Madrid"): cae en
# el prior de Alpaca (69,5% del corpus) de que toda pregunta personal tiene
# respuesta. El "[Nombre Completo]" viene de los 586 ejemplos de Alpaca con
# placeholders tipo formulario.
# FIX: +944 ejemplos NUEVOS de sondas de identidad (580 ES / 364 EN, ~40
# formas por sonda, respuestas ancladas en Alfa1/Minaya AI), a x1. NO es
# oversample: es cobertura que hoy es 0. Nada mas cambia vs v16. TOTAL 90.286.
#
# RESULTADO Exp.21: val loss 1.1808 (el mas bajo). control_identidad 35 -> 39/40
# (identidad_es 15 -> 20/20 perfecto: id07 "¿donde vives?" y id04/id06
# arreglados en espanol). Seguridad ~intacta. El "techo de 520M" para
# identidad era FALSO: era cobertura. Ver scoring_v16_vs_v21.md.
# Unico defecto: id_en10 metia cola de ciberseguridad.
#
# EXPERIMENTO 22 (2026-09-08): identidad_cobertura v2 (+sonda "apagado") +
# titulo_cobertura (738). RESULTADO: apagado ARREGLA id_en10 (identidad_en
# 20/20). Pero titulo_cobertura = +4 en general_titulo / -4 en
# agente_titulador (alucina "Ventura:", "en Sudamerica" en titulares de
# noticia). Titulo es limite de CAPACIDAD (synthesize-vs-trim de 520M), no
# hueco de cobertura. titulo_cobertura DESCARTADO. Ver scoring_v16_vs_v22.md.
#
# EXPERIMENTO 23 (2026-09-08): v16 + identidad_cobertura (CON apagado), SIN
# titulo_cobertura. Objetivo: +4 identidad + id_en10 arreglado, sin la
# regresion de agente_titulador. Unico cambio vs v16: 1008 ejemplos de
# sondas de identidad (612 ES / 396 EN, con apagado), a x1. Candidato a
# nueva referencia. TOTAL ~90.350.
#
# EXPERIMENTO 24 (2026-09-11): corpus de v23 + 3 generadores nuevos dirigidos
# a agujeros de cobertura DETECTADOS EN EL EXPERIMENTO DE CONTROL con
# Qwen2.5-0.5B-base (mismo corpus, full fine-tune aparte, ver
# Benchmark_SFT/mtbench/SCORING_mtbench_v2_vs_v1.md): al comparar el SFT
# propio contra Qwen-Instruct en el MT-Bench bilingue oficial se vio que el
# fallo mas grave (9/80 EN y 8/80 ES de respuestas VACIAS en el segundo turno
# de conversaciones tipo "reescribe/critica tu respuesta anterior") no era un
# bug de sampling (confirmado con test de aislamiento variando temperatura/
# rep_penalty/greedy/eos: 0/9 no-vacias en TODAS las configs) sino cobertura
# real: 0/90.350 ejemplos del corpus tenian esa forma estructural. Se
# construyeron y auditaron 3 generadores nuevos (D:/Generar_SFT/
# refuerzo_dominios/): generar_multiturno_estructurado.py (1.808: T1 codigo/
# JSON/tabla + T2 transformacion sobre la misma respuesta), generar_
# restricciones_formato.py (352: acrostico, N palabras exactas, sin verbos,
# mayusculas, limite de caracteres, una frase), generar_roleplay.py (104:
# 13 personas profesionales genericas con T2 de seguimiento). +2.264 ejemplos
# verificados (autotest/verifica() antes de emitir en los 3). TOTAL 93.963.
# En el control con Qwen se confirmo que el ataque a estos 3 huecos SI resuelve
# el sintoma objetivo (vacias EN 9->1, ES 8->2; longitud media T2 +30%) sin
# tocar el codigo de inferencia -- validando que era cobertura, no bug, antes
# de gastar un ciclo de entreno completo del 520M en la hipotesis equivocada.
# EXPERIMENTO 25 (2026-09-11): v24 + conversacion_libre (128 items, x1).
# Objetivo: atacar el colapso en conversacion libre de 3+ turnos que Exp.24
# NO tocaba -- confirmado en vivo (ejemplo real del usuario, viaje/alerta
# alimentaria: turno de seguimiento devolvio la respuesta anterior palabra
# por palabra) y en el replay del log real: 32 hallazgos en 536 turnos,
# MISMO TOTAL en v23 y v24 pero 16 casos DISTINTOS en cada uno (prueba de
# que Exp.24 no atacaba esta causa raiz). Dos patrones nuevos: (A)
# seguimiento_tema (20 cadenas de 3 turnos, respuesta especifica y distinta
# en cada turno) y (B) identidad_tras_contexto (108 items: 3-5 turnos de
# relleno + sonda de identidad, respuesta final siempre del banco anclado).
# Ver D:/Generar_SFT/refuerzo_dominios/generar_conversacion_libre.py.
# TOTAL 94.091.
# EXPERIMENTO 26 (2026-09-11): v25 + seguimiento_tema (568 items = 142 x4).
# Ataca el patron de fallo DOMINANTE del replay: 56% de los ~17 colapsos
# reales/corrida son "seguimiento sobre el mismo tema -> repite la respuesta
# anterior" (categorizado sobre 6 corridas de varianza tras arreglar el
# detector, que tenia falsos positivos de identidad). NO confundir con
# conversacion_libre (Exp.25, 128 items, 0,14%, no movio nada) ni con
# multiturno_extendido (cuya plantilla de 8 frases-acuse resulto ser el 0%
# de los colapsos medidos, comprobado antes de tocarla). 3 subpatrones:
# polisemia (palabra polisemica NATIVA por idioma, no traducida),
# peticion incremental, seguimiento de tema encadenado. x4 -> ~0,6% del
# corpus. Ver generar_seguimiento_tema.py y analisis_corpus_2026-09-11.md.
# Metrica: replay con >=3 corridas (v24 baseline: 17,7; v25: 17,0). TOTAL 94.659.
# v25 y v26 evaluados pero NO adoptados (indistinguibles de v24) -- v24 sigue
# siendo la referencia vigente (D:\alpha_1_05B_it_v24.pt).
#
# EXPERIMENTO 27 (2026-09-12): v26 + agente_react_es (806 ejemplos, x1,
# D:/Generar_SFT/agente_react/), primer dataset de bucle de agente ReAct
# (Thought/Action/Observation, buscar[] real via BM25 + calcular[] real via
# AST) para atacar el fallo documentado en alfa_sft_agent/README.md:
# agente_tool_use en 0/2 en tres experimentos previos (formato Hermes-style
# sin razonamiento explicito, el modelo acertaba la sintaxis pero fallaba al
# elegir herramienta y al encadenar el resultado de una accion con la
# siguiente). Las Observations del corpus se ejecutaron de verdad (nunca las
# escribe el LLM generador); ver herramientas_react.py para el detalle. Banco
# de evaluacion held-out de 87 ejemplos (react_agente_es_eval.jsonl) queda
# FUERA de este directorio a proposito, para evaluar despues con
# evaluar_agente.py (--backend ollama) sin haber visto esos ejemplos en
# entreno. TOTAL 95.465.
# RESULTADO Exp.27: val loss 1.1195 (mejor que v24 1.1372). Eval ReAct held-out
# (ollama): calcular_simple 53% correcto, buscar_y_calcular mayoria fallo,
# buscar_factual 33% abstencion. Diagnostico: la cadena buscar+calcular tenia
# solo 17 ejemplos de entreno (vs 400/389) y el prompt-plantilla se copiaba
# literal en un 520M. NO adoptado; se itera en Exp.28.
#
# EXPERIMENTO 28 (2026-09-13): mismas correcciones dirigidas al diagnostico
# del Exp.27, en D:/Generar_SFT/agente_react/:
#  (a) cadena buscar+calcular de 17 -> ~200 ejemplos: ENTIDADES_FECHA_TRAIN
#      ampliado de 18 a 70 entidades curadas (todas con hecho verificado en
#      herramientas_react.py, disjuntas de las 30 de eval) y 3 plantillas de
#      fraseo por entidad.
#  (b) prompt INSTRUCCIONES reescrito en prosa, SIN el bloque-plantilla literal
#      "o bien (para calcular):" que el modelo copiaba palabra por palabra.
#  (c) RE_ACTION sin ancla de fin de linea, para capturar acciones aunque el
#      modelo continue escribiendo tras el corchete ("calcular[..] = X").
# RESULTADO Exp.30: val loss 1.1182. Estado KNOW arreglado: busqueda
# innecesaria 90%->10%, acierto directo 0%->65% (banco_calibracion_busqueda.
# jsonl). Sin regresion (seguridad 10/10, identidad 20/20, JSON 12/12); la
# regresion social de v29 se recupero sola. ADOPTADO como referencia
# (alpha1-it/alfa-it en Ollama, D:\alpha_1_05B_it_v30.pt) tras investigar y
# descartar como ruido un hallazgo puntual (gc01 "56/10", 0/12 reproducido).
#
# EXPERIMENTO 31 (2026-09-13): v30 + dos correcciones sobre fallos REALES
# encontrados probando v30 en chat_web_app.py (modo agente añadido ese
# mismo dia, ver logs_chat_web/chat_log_2026-09-13.jsonl):
#  (a) contraste_know (50 items): responder_directo (v30) enseño la FORMA
#      "pregunta corta -> responde directo" pero no afilo el limite de
#      fiabilidad -- v30 alucino con confianza "Quien escribio La Celestina?"
#      -> "Fernando the Blind" (inventado) y broke "Quien fue Blas de Lezo?"
#      atribuyendole la Guerra de Independencia (imposible, murio en 1741,
#      ~70 anios antes). Estas 50 preguntas tienen la MISMA forma superficial
#      que QA_CANONICAS (quien escribio/pinto/fundo X) pero de figuras menos
#      icónicas, procesadas con busqueda real (ejecutar_react) para que el
#      contraste sea explicito, no solo mas volumen de la misma forma. Se quito
#      "La Regenta" de QA_CANONICAS (v30) por ambiguedad -- el uso real ya la
#      trataba como incierta -- para no contradecir el contraste.
#  (b) sin_herramienta (40 items): con el prompt ReAct activo, tareas NO
#      factuales (escribir un email, dar pasos de una receta) forzaban una
#      Action sin sentido -- "escribe un email a mi jefe" genero
#      buscar[que hora no puedo ir al trabajo], trajo un fragmento sobre un
#      album de musica, y redacto el email mezclando esa basura. Trazas de UN
#      turno sin Action, respuesta generada por el LLM de referencia SIN el
#      prompt ReAct (para no degradar la calidad de redaccion). TOTAL ~96.240.
# v42a ADOPTADO (2026-09-23). v42b (AlpaGasus en lugar de Alpaca) se entreno,
# se midio en los tres bancos y se DESCARTO: MT-Bench ES 1,39 / EN 1,55 frente
# a 1,67 / 1,69 de v42a, con IFEval tambien algo por debajo. Su corpus y sus
# checkpoints se borraron del disco; las MEDICIONES se conservan. Aviso: el
# val loss de v42b (1,107) era mejor que el de v42a (1,253) y NO significaba
# nada -- el split de validacion cambio entero al cambiar la fuente mayoritaria.
CHECKPOINT_PATH = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints\alpha_1_sft_release_v43.pt"
# CHECKPOINT_PATH = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints\alpha_1_sft_release_v41.pt"
# Exp.32b (2026-09-14): CORRECCION de Exp.32. v32 (con extraccion_real)
# REGRESIONO el enrutado KNOW/SEARCH: en el banco de calibracion (3 corridas
# T=0.2 vs v30) "busco_sin_necesidad" subio de 10% a 25% y KNOW correcto bajo
# de 68% a 53% -- mismo fallo que Exp.31 (ver memoria [[react-agente-sft-hallazgos]]).
# Las 20 trazas de extraccion_real SIEMPRE buscan y desplazan la frontera fina
# KNOW/SEARCH hacia sobre-busqueda. v32b RETIRA extraccion_real y conserva solo
# pregunta_test (7 ej.) + cuento_protagonista (15 ej.), que son redaccion plana
# sin ReAct y NO tocan el enrutado. Corpus = v30 + 22 ej. de redaccion.
# Ver memoria [[dataset-v32-nuevas-categorias]].
# ANTERIOR (Exp.28): Banco de eval held-out SIN CAMBIOS (react_agente_es_eval.
# jsonl, mismas 87 preguntas) para comparar v27 vs v28 limpio. TOTAL ~95.850.
# RESULTADO Exp.28: val loss 1.1143 (récord). Eval ReAct: calcular_simple
# 53%->87%, fallos de formato 25->4. PERO la cadena buscar+calcular pasó a
# fabricar años (89% incorrecto): al completar el bucle con búsqueda real
# fallida (entidades held-out sin hecho curado, BM25 no las recupera), en vez
# de abstenerse inventaba una cifra. Regresión suite 316: seguridad 10/10 e
# identidad 20/20 intactas, JSON 12/12, sin colapso. NO adoptado; se itera v29.
#
# EXPERIMENTO 29 (2026-09-13): v28 + cadena-abstención. Añade ~120 trazas
# programáticas de cadena donde la búsqueda NO devuelve el año pedido
# (entidades ficticias; Observation = salida REAL de buscar(), a menudo un
# fragmento irrelevante CON años señuelo) y la respuesta correcta es
# abstenerse. Cierra el hueco del Exp.28: el modelo nunca había visto una
# cadena con búsqueda fallida, por eso fabricaba. evaluar_agente.py ahora
# cuenta la abstención honesta como acierto en la cadena held-out (mide al
# modelo, no al BM25 que no puede recuperar esas fechas). Eval held-out SIN
# CAMBIOS. TOTAL ~95.970.
# RESULTADO Exp.29: val loss 1.1231. Cadena: fabricación de años 89%->11%,
# acierto honesto (correcto+abstención) ~4%->82%, calcular_simple 87% intacto.
# Seguridad 10/10, identidad 20/20, JSON 12/12. Regresión menor y acotada en
# general_social ES (gs04). NO adoptado; se itera v30.
#
# EXPERIMENTO 30 (2026-09-13): v29 + responder_directo. Añade ~180 trazas de
# UN turno (sin Action) a preguntas de conocimiento canónico (capital de X,
# quién escribió/pintó una obra muy famosa, ciencia básica), construidas de
# 85 QA curadas x 3 formulaciones de Thought que explicitan el discriminador
# "dato muy conocido -> no busco". Cierra el hueco medido con
# banco_calibracion_busqueda.jsonl sobre v29: el modelo usaba buscar[] hasta
# para "¿capital de España?" (90%) porque NINGUNA traza de entreno respondía
# sin herramienta. QA disjuntas del banco de calibración. TOTAL ~96.150.
# (CHECKPOINT_PATH real de este experimento: ver arriba, antes del bloque de
# Exp.31 -- se movio ahi para que la asignacion activa sea siempre la ULTIMA
# experimento, no esta linea historica.)
SEED = 1337

# --- Generacion de prueba ---
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_K = 40

