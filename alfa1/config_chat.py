# sancho/config_chat.py

# Checkpoint
USE_BEST_CHECKPOINT = True

# Generación

MAX_NEW_TOKENS_CHAT = 300   # tokens máximos por respuesta
TEMPERATURE_CHAT = 0.10     # AJUSTAR: más alto = más variedad, más riesgo de divagar
TOP_K_CHAT = 30             # AJUSTAR: nº de tokens candidatos en cada paso de sampling

REPETITION_PENALTY = 1.30   # AJUSTAR: 1.0 = sin penalización; sube si repite frases/párrafos
REPETITION_WINDOW = 60      # AJUSTAR: nº de tokens generados recientes a penalizar (0 = toda la secuencia)

ASSISTANT_NAME = "Alfa"
USER_NAME = "Tú"


# Historial
USE_HISTORY = False
MAX_HISTORY_TURNS = 2



# Debug
SHOW_PROMPT_DEBUG = False
SHOW_TOKEN_COUNT = False
