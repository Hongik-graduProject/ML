from __future__ import annotations

from app.config import get_settings
from app.services.artifact_store import ArtifactStore


def main() -> None:
    settings = get_settings()
    store = ArtifactStore(settings)
    store.load()
    print(f"OK: model={settings.model_path.name}")
    print(f"OK: features={len(store.features)}")
    print(f"OK: predictions={len(store.predictions)}")
    print(f"OK: latest_date={store.predictions['Date'].max().date()}")


if __name__ == "__main__":
    main()
