"""SVC focused v2 — binary per-topic classifier, 8 topics, CV-honest mixture."""
RANDOM_STATE = 42
EXCLUDED_TOPICS = {"Ковид", "Другое", "НПС"}
ALLOWED_TOPICS = [
    "Защита прав потребителей",
    "ДКП в мире",
    "ДКП",
    "Финансовый рынок",
    "НБП",
    "Финтех",
    "Геополитика",
    "НДО",
]
PRIORITY_TOPICS = ["НБП", "Защита прав потребителей", "Геополитика"]

# Map from raw data column names to canonical ALLOWED_TOPICS display names.
# Data files use short aliases; ALLOWED_TOPICS uses full display names.
TOPIC_COL_MAP = {
    "ЗПП и ФГН": "Защита прав потребителей",
    "ДКП в мире": "ДКП в мире",
    "ДКП": "ДКП",
    "Финрынки и банки": "Финансовый рынок",
    "НБП": "НБП",
    "Финтех": "Финтех",
    "Геополитика": "Геополитика",
    "НДО": "НДО",
    # Excluded — not in ALLOWED_TOPICS, listed here for completeness
    "Другое": None,
    "Ковид": None,
    "НПС": None,
}
# Reverse map: canonical name → data column name
COL_TOPIC_MAP = {v: k for k, v in TOPIC_COL_MAP.items() if v is not None}
