"""
Script para descargar imágenes de guitarras para clasificación ML.
Usa DuckDuckGo Image Search (sin API key requerida).

Instalación de dependencias:
    pip install ddgs requests Pillow tqdm

Uso:
    python download_guitars.py
    python download_guitars.py --images 150 --output ./mi_dataset
    python download_guitars.py --images 200 --output ./dataset
"""

import os
import time
import random
import hashlib
import argparse
import requests
from pathlib import Path
from io import BytesIO

# ── Importar librería (soporta nombre viejo y nuevo) ──────────────────────────
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError(
            "Instala la librería: pip install ddgs\n"
            "(El paquete duckduckgo_search fue renombrado a ddgs)"
        )

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("⚠️  Pillow no instalado. No se validarán imágenes. Instala con: pip install Pillow")
    PIL_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ──────────────────────────────────────────────
# Configuración de clases y queries de búsqueda
# ──────────────────────────────────────────────

GUITAR_CLASSES = {
    "fender_stratocaster": [
        "Fender Stratocaster electric guitar",
        "Fender Strat sunburst",
        "Fender Stratocaster American Standard",
        "Fender Player Stratocaster",
    ],
    "fender_telecaster": [
        "Fender Telecaster electric guitar",
        "Fender Tele guitar",
        "Fender American Telecaster",
        "Fender Player Telecaster",
    ],
    "fender_jaguar": [
        "Fender Jaguar electric guitar",
        "Fender Jaguar offset guitar",
        "Fender Classic Player Jaguar",
        "Fender Jaguar HH",
    ],
    "gibson_les_paul_standard": [
        "Gibson Les Paul Standard electric guitar",
        "Gibson Les Paul Standard sunburst",
        "Gibson Les Paul Standard gold top",
        "Gibson Les Paul Standard 50s",
    ],
    "gibson_les_paul_studio": [
        "Gibson Les Paul Studio electric guitar",
        "Gibson Les Paul Studio worn",
        "Gibson Les Paul Studio ebony",
        "Gibson Les Paul Studio guitar",
    ],
    "ibanez": [
        "Ibanez electric guitar",
        "Ibanez RG series",
        "Ibanez AZ guitar",
        "Ibanez Prestige guitar",
    ],
}

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────

DEFAULT_IMAGES_PER_CLASS = 100
DEFAULT_OUTPUT_DIR = "./guitar_dataset"
MIN_WIDTH = 150
MIN_HEIGHT = 150
REQUEST_TIMEOUT = 10

# Delays para evitar rate limiting de DuckDuckGo
DELAY_BETWEEN_DOWNLOADS = 0.5    # entre cada descarga de imagen
DELAY_BETWEEN_QUERIES   = 4.0    # entre cada query de búsqueda
DELAY_BETWEEN_CLASSES   = 8.0    # entre cada clase
DELAY_JITTER            = 2.0    # variación aleatoria adicional


def parse_args():
    parser = argparse.ArgumentParser(description="Descarga imágenes de guitarras para ML")
    parser.add_argument("--images", type=int, default=DEFAULT_IMAGES_PER_CLASS,
                        help=f"Imágenes por clase (default: {DEFAULT_IMAGES_PER_CLASS})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})")
    return parser.parse_args()


def sleep_with_jitter(base: float, jitter: float = DELAY_JITTER):
    """Sleep con variación aleatoria para parecer tráfico humano."""
    time.sleep(base + random.uniform(0, jitter))


def create_dirs(base_path: Path):
    for class_name in GUITAR_CLASSES:
        (base_path / class_name).mkdir(parents=True, exist_ok=True)
    print(f"📁 Dataset en: {base_path.resolve()}\n")


def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def is_valid_image(data: bytes) -> bool:
    if not PIL_AVAILABLE:
        return len(data) > 5_000
    try:
        img = Image.open(BytesIO(data))
        img.verify()
        img = Image.open(BytesIO(data))
        w, h = img.size
        return w >= MIN_WIDTH and h >= MIN_HEIGHT
    except Exception:
        return False


def fetch_image_urls(query: str, max_results: int, retries: int = 3) -> list:
    """Obtiene URLs con reintentos y backoff exponencial ante rate limits."""
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=max_results))
            urls = [r["image"] for r in results if "image" in r]
            return urls
        except Exception as e:
            err_str = str(e)
            if "403" in err_str or "Ratelimit" in err_str or "429" in err_str:
                wait = (attempt + 1) * 15 + random.uniform(0, 5)
                print(f"   ⏳ Rate limit — esperando {wait:.0f}s antes de reintentar "
                      f"(intento {attempt + 1}/{retries})...")
                time.sleep(wait)
            else:
                print(f"   ⚠️  Error buscando '{query}': {e}")
                break
    return []


def download_image(url: str, dest_path: Path) -> bool:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not any(
            url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")
        ):
            return False

        data = resp.content
        if not is_valid_image(data):
            return False

        if "png" in content_type or url.lower().endswith(".png"):
            ext = ".png"
        elif "webp" in content_type or url.lower().endswith(".webp"):
            ext = ".webp"
        else:
            ext = ".jpg"

        dest_path.with_suffix(ext).write_bytes(data)
        return True
    except Exception:
        return False


def download_class(class_name: str, queries: list, base_path: Path, target: int):
    class_dir = base_path / class_name
    existing = len(list(class_dir.glob("*.*")))
    needed = target - existing

    if needed <= 0:
        print(f"  ✅ {class_name}: ya tiene {existing} imágenes, se omite.")
        return

    print(f"\n{'─'*55}")
    print(f"  🎸 {class_name.replace('_', ' ').upper()}")
    print(f"  Existentes: {existing} | Necesarias: {needed}")
    print(f"{'─'*55}")

    downloaded = 0
    seen_urls: set = set()
    per_query = max(int(needed * 1.6 / len(queries)), 25)

    for i, query in enumerate(queries):
        if downloaded >= needed:
            break

        if i > 0:
            print(f"  ⏳ Pausa entre queries ({DELAY_BETWEEN_QUERIES:.0f}s)...")
            sleep_with_jitter(DELAY_BETWEEN_QUERIES)

        print(f"  🔍 Buscando: \"{query}\"")
        urls = fetch_image_urls(query, per_query)

        if not urls:
            print(f"      → Sin resultados, continuando...")
            continue

        print(f"      → {len(urls)} URLs encontradas")

        iterator = tqdm(urls, desc="  Descargando", unit="img") if TQDM_AVAILABLE else urls

        for url in iterator:
            if downloaded >= needed:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            uid = hash_url(url)
            idx = existing + downloaded + 1
            dest = class_dir / f"{class_name}_{idx:04d}_{uid}"

            if download_image(url, dest):
                downloaded += 1

            sleep_with_jitter(DELAY_BETWEEN_DOWNLOADS, jitter=0.5)

    total = existing + downloaded
    print(f"  ✔  Descargadas {downloaded} nuevas | Total en clase: {total}")


def print_summary(base_path: Path):
    print(f"\n{'═'*55}")
    print("  📊 RESUMEN DEL DATASET")
    print(f"{'═'*55}")
    grand_total = 0
    for class_name in GUITAR_CLASSES:
        count = len(list((base_path / class_name).glob("*.*")))
        grand_total += count
        bar = "█" * (count // 5)
        print(f"  {class_name:<30} {count:>4}  {bar}")
    print(f"{'─'*55}")
    print(f"  {'TOTAL':<30} {grand_total:>4}")
    print(f"{'═'*55}\n")
    print(f"  📁 Ubicación: {base_path.resolve()}")
    print("  💡 Listo para usar con:")
    print("     torchvision.datasets.ImageFolder")
    print("     tf.keras.utils.image_dataset_from_directory\n")


def main():
    args = parse_args()
    base_path = Path(args.output)

    print("\n" + "═" * 55)
    print("  🎸 GUITAR DATASET DOWNLOADER")
    print("═" * 55)
    print(f"  Clases       : {len(GUITAR_CLASSES)}")
    print(f"  Meta/clase   : {args.images} imágenes")
    print(f"  Total aprox. : {len(GUITAR_CLASSES) * args.images} imágenes")
    print(f"  Directorio   : {base_path.resolve()}")
    print("═" * 55 + "\n")

    create_dirs(base_path)

    for idx, (class_name, queries) in enumerate(GUITAR_CLASSES.items()):
        if idx > 0:
            print(f"\n  ⏳ Pausa entre clases ({DELAY_BETWEEN_CLASSES:.0f}s)...")
            sleep_with_jitter(DELAY_BETWEEN_CLASSES)

        download_class(class_name, queries, base_path, args.images)

    print_summary(base_path)


if __name__ == "__main__":
    main()
