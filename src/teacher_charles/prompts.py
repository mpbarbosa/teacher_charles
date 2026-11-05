"""Carregamento dos prompts de ``prompts/``.

Os prompts são arquivos ``.md`` versionados, não strings no código Python:
ajustar a persona não deve exigir mexer em Python, e o diff de uma mudança de
prompt tem que ser legível no git.

O cache invalida por mtime, então editar um ``.md`` vale no próximo turno sem
reiniciar o servidor.
"""

from __future__ import annotations

from pathlib import Path


class PromptNotFound(FileNotFoundError):
    pass


class PromptLibrary:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, tuple[float, str]] = {}

    def path_for(self, name: str) -> Path:
        """``"persona"`` -> ``prompts/persona.md``; ``"modes/drill"`` também serve."""
        if name.endswith(".md"):
            name = name[:-3]
        candidate = (self.root / f"{name}.md").resolve()
        # Um nome com ".." sairia de prompts/; os nomes são internos, mas a
        # checagem é de uma linha e remove a classe de erro inteira.
        if not candidate.is_relative_to(self.root.resolve()):
            raise PromptNotFound(f"nome de prompt fora de prompts/: {name!r}")
        return candidate

    def get(self, name: str) -> str:
        path = self.path_for(name)
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            raise PromptNotFound(
                f"prompt {name!r} não existe em {self.root}. Esperado: {path}"
            ) from exc

        cached = self._cache.get(name)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        text = path.read_text(encoding="utf-8").strip()
        self._cache[name] = (mtime, text)
        return text

    def exists(self, name: str) -> bool:
        try:
            return self.path_for(name).is_file()
        except PromptNotFound:
            return False
