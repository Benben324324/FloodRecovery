"""Download the project's public inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.download import download_ntl, download_context
from src.radar import prepare_radar

if __name__ == '__main__':
    print(download_ntl())
    download_context()
    print(prepare_radar())
