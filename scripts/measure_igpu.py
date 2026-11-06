#!/usr/bin/env python3
"""Mede a iGPU Intel Arc contra a CPU pura, com o mesmo modelo.

A §1.3 do plano da Fase 1 mediu 9,3 tok/s num 3B Q4 e registrou a causa:
``/api/ps`` reportava ``size_vram=0.00GB`` — o Ollama rodava 100% em CPU e a
Arc integrada do Meteor Lake estava ociosa. Ficou anotada como **a** alavanca
para o turno de voz de 5,4-7,4s da §3.1 do plano 1.5.

Este script abre a alavanca e mede o que ela entrega. O Ollama 0.35 já traz
``libggml-vulkan.so``; ``OLLAMA_VULKAN=1`` e ``OLLAMA_IGPU_ENABLE=1`` ligam o
backend sem compilar nada:

    OLLAMA_HOST=127.0.0.1:11435 OLLAMA_VULKAN=1 OLLAMA_IGPU_ENABLE=1 \
        ollama serve

    uv run python scripts/measure_igpu.py
        Compara os dois servidores no modelo de ``TC_LLM_MODEL``: geração e
        avaliação de prompt, pelo ``eval_count``/``eval_duration`` do próprio
        servidor, que é como a §1.3 mediu os 9,3 tok/s.

    uv run python scripts/measure_igpu.py --model qwen2.5:7b-instruct-q4_K_M
        O mesmo no 7B, contra os 4,0 tok/s da §1.3.

    uv run python scripts/measure_igpu.py --reps 5
        Mais repetições; o relatório usa a mediana.

## Por que "compilou com Vulkan" não basta

Duas confirmações independentes de que a iGPU está de fato trabalhando, as
duas sem root (o ``intel_gpu_top`` desta máquina pede ``CAP_PERFMON``):

- ``size_vram`` do ``/api/ps``: quanto do modelo o servidor diz ter posto na
  GPU. Era 0,00 GB na §1.3.
- ``drm-engine-render`` do ``/proc/<pid>/fdinfo/<fd>`` do ``llama-server``: o
  tempo que o motor de render do i915 passou ocupado. Dividido pelo tempo de
  parede da chamada, dá a ocupação da GPU durante a geração.

Precisa dos dois servidores de pé. Sem o da iGPU, o script mede só a CPU e diz
o que falta.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import httpx  # noqa: E402

from teacher_charles.config import get_settings  # noqa: E402

CPU_URL = "http://127.0.0.1:11434"
IGPU_URL = "http://127.0.0.1:11435"

# Prompt curto, para medir geração sem o prompt dominar a conta. Temperatura 0
# e um pedido fechado para as duas pontas gerarem a mesma coisa.
GEN_PROMPT = "Write one paragraph about the sea. Plain prose, no lists."

# Prompt longo, para medir avaliação de prompt com amostra que presta: 37
# tokens dão um número com ruído maior que o efeito que se quer ver.
# Uma frase de aluno, repetida, para montar prompt de um tamanho pedido. Mede
# o que o produto faz: prosa em inglês com o erro que a base contrastiva pega.
_FILLER = "The student said: I have 30 years and I am living here since 2019. "

# Os tamanhos que decidem. 40 tokens é o ponto que a §1.3 mediu (os 54 tok/s);
# 700 é a ordem do prefixo estável do turno (persona + nível + modo, ~674
# tokens pelo CLAUDE.md); 1400 é o turno com histórico.
PROMPT_SIZES = (40, 700, 1400)


def prompt_of(target_tokens: int) -> str:
    """Prompt de ~``target_tokens`` tokens, com nonce **na frente**.

    O nonce vai na frente, e isto não é estilo: o Ollama cacheia o *prefixo*
    do prompt. Com o nonce no fim, a segunda rodada reaproveita tudo menos o
    rabo, e a conta vira `prompt_eval_count` de 5 tokens sobre 2 ms — saem
    2500 tok/s que não medem avaliação de prompt nenhuma.
    """
    # ~15 tokens por repetição do filler, medido no tokenizador do qwen2.5.
    repeats = max(1, round(target_tokens / 15))
    return f"[{time.time_ns()}] " + _FILLER * repeats


GEN_TOKENS = 150


class GpuWatch:
    """Amostra o ``fdinfo`` do i915 e a frequência da GT durante uma chamada.

    O ``llama-server`` vira e mexe é reiniciado pelo Ollama (troca de modelo,
    expiração), então o PID é descoberto a cada medição, não guardado.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.busy_ns = 0
        self.wall_s = 0.0
        self.freq_mhz: list[int] = []
        self.resident_kib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _serve_pid(url: str) -> int:
        """O PID do ``ollama serve`` que atende esta URL, pelo environ.

        Com dois servidores na máquina — um de CPU, um de Vulkan — somar o
        ``fdinfo`` dos dois atribuiria ao servidor medido o trabalho do outro.
        """
        host = url.removeprefix("http://").removeprefix("https://").rstrip("/")
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                cmdline = (proc / "cmdline").read_bytes()
                if b"ollama" not in cmdline or b"serve" not in cmdline:
                    continue
                environ = (proc / "environ").read_bytes().decode(errors="replace")
            except OSError:
                continue
            declared = next(
                (
                    item.removeprefix("OLLAMA_HOST=")
                    for item in environ.split("\0")
                    if item.startswith("OLLAMA_HOST=")
                ),
                "",
            )
            # Sem `OLLAMA_HOST` no environ o servidor está no default — e é o
            # caso do servidor que o aluno já tem de pé. Sem este ramo o
            # casamento falha, o escopo cai para "todo llama-server da
            # máquina" e o trabalho de um servidor é contado no outro.
            if not declared:
                declared = "127.0.0.1:11434"
            if host in declared or declared in host:
                return int(proc.name)
        return 0

    @classmethod
    def _render_fds(cls, url: str) -> list[Path]:
        """Os fds de DRM dos ``llama-server`` deste servidor, e só dele."""
        serve_pid = cls._serve_pid(url)
        fds: list[Path] = []
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                cmdline = (proc / "cmdline").read_bytes()
            except OSError:
                continue
            if b"llama-server" not in cmdline:
                continue
            if serve_pid:
                try:
                    stat = (proc / "stat").read_text()
                    ppid = int(stat.rsplit(")", 1)[1].split()[1])
                except (OSError, IndexError, ValueError):
                    continue
                if ppid != serve_pid:
                    continue
            try:
                entries = list((proc / "fdinfo").iterdir())
            except OSError:
                continue
            for fd in entries:
                try:
                    text = fd.read_text()
                except OSError:
                    continue
                if "drm-engine-render" in text:
                    fds.append(fd)
        return fds

    @staticmethod
    def _read(fds: list[Path]) -> tuple[int, int]:
        busy = resident = 0
        for fd in fds:
            try:
                text = fd.read_text()
            except OSError:
                continue
            for line in text.splitlines():
                if line.startswith("drm-engine-render:"):
                    busy += int(line.split()[1])
                elif line.startswith("drm-resident-system0:"):
                    resident += int(line.split()[1])
        return busy, resident

    @staticmethod
    def _freq() -> int:
        for path in (
            "/sys/class/drm/card1/gt_act_freq_mhz",
            "/sys/class/drm/card1/gt/gt0/rps_act_freq_mhz",
        ):
            try:
                return int(Path(path).read_text().strip())
            except (OSError, ValueError):
                continue
        return 0

    def _loop(self, fds: list[Path]) -> None:
        while not self._stop.wait(0.25):
            self.freq_mhz.append(self._freq())
            _, resident = self._read(fds)
            self.resident_kib = max(self.resident_kib, resident)

    def __enter__(self) -> GpuWatch:
        self._fds = self._render_fds(self.url)
        self._busy0, self.resident_kib = self._read(self._fds)
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, args=(self._fds,), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.wall_s = time.perf_counter() - self._t0
        busy1, _ = self._read(self._fds)
        self.busy_ns = busy1 - self._busy0

    @property
    def busy_pct(self) -> float:
        if self.wall_s <= 0:
            return 0.0
        return 100.0 * (self.busy_ns / 1e9) / self.wall_s

    @property
    def peak_freq(self) -> int:
        return max(self.freq_mhz, default=0)


def generate(url: str, model: str, prompt: str, num_predict: int) -> dict:
    with httpx.Client(timeout=900.0) as client:
        reply = client.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": num_predict, "temperature": 0},
            },
        )
        reply.raise_for_status()
        return reply.json()


def unload(url: str, model: str) -> None:
    """Expira o modelo, para a próxima carga ser honesta."""
    try:
        with httpx.Client(timeout=60.0) as client:
            client.post(
                f"{url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
            )
    except httpx.HTTPError:
        pass


def vram_gb(url: str, model: str) -> float:
    try:
        with httpx.Client(timeout=15.0) as client:
            running = client.get(f"{url}/api/ps").json()
    except httpx.HTTPError:
        return -1.0
    for entry in running.get("models", []):
        if entry.get("name") == model:
            return entry.get("size_vram", 0) / 1e9
    return -1.0


def alive(url: str) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            return client.get(f"{url}/api/version").status_code == 200
    except httpx.HTTPError:
        return False


def quiesce(url: str, keep: str) -> None:
    """Descarrega todo modelo exceto o que vai ser medido.

    Sem isto a comparação é injusta: um 7B parado na RAM do servidor da CPU
    disputa banda de memória com o 3B que está sendo medido.
    """
    try:
        with httpx.Client(timeout=15.0) as client:
            for entry in client.get(f"{url}/api/ps").json().get("models", []):
                name = entry.get("name", "")
                if name and name != keep:
                    unload(url, name)
    except httpx.HTTPError:
        pass


def measure(url: str, model: str, reps: int) -> dict:
    quiesce(url, model)
    # Aquece: a carga do modelo não entra em nenhum número de vazão.
    generate(url, model, "hi", 4)

    gen_rates: list[float] = []
    watches: list[GpuWatch] = []

    for _ in range(reps):
        with GpuWatch(url) as watch:
            data = generate(url, model, GEN_PROMPT, GEN_TOKENS)
        watches.append(watch)
        if data.get("eval_duration"):
            gen_rates.append(data["eval_count"] / (data["eval_duration"] / 1e9))

    # Avaliação de prompt, em curva: 1 token gerado, prompt novo a cada rodada.
    prompt_curve: dict[int, dict] = {}
    for size in PROMPT_SIZES:
        rates: list[float] = []
        counts: list[int] = []
        for _ in range(reps):
            data = generate(url, model, prompt_of(size), 1)
            if data.get("prompt_eval_duration"):
                counts.append(data["prompt_eval_count"])
                rates.append(data["prompt_eval_count"] / (data["prompt_eval_duration"] / 1e9))
        prompt_curve[size] = {
            "tok_s": statistics.median(rates) if rates else 0.0,
            "tokens": statistics.median(counts) if counts else 0,
            "samples": [round(r, 1) for r in rates],
        }

    best = max(watches, key=lambda w: w.busy_pct)
    return {
        "gen_tok_s": statistics.median(gen_rates) if gen_rates else 0.0,
        "gen_samples": [round(r, 1) for r in gen_rates],
        "prompt_curve": prompt_curve,
        "size_vram_gb": vram_gb(url, model),
        "gpu_busy_pct": best.busy_pct,
        "gpu_peak_mhz": best.peak_freq,
        "gpu_resident_gb": best.resident_kib / 1e6,
    }


def report(label: str, result: dict) -> None:
    print(f"\n### {label}")
    print(f"  geração           {result['gen_tok_s']:.1f} tok/s   {result['gen_samples']}")
    for size, entry in result["prompt_curve"].items():
        print(
            f"  prompt ~{size:>4}     {entry['tok_s']:7.1f} tok/s"
            f"   ({entry['tokens']:.0f} tok)   {entry['samples']}"
        )
    print(f"  size_vram         {result['size_vram_gb']:.2f} GB")
    print(
        f"  iGPU render busy  {result['gpu_busy_pct']:.0f}%"
        f"   pico {result['gpu_peak_mhz']} MHz"
        f"   residente {result['gpu_resident_gb']:.2f} GB"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="default: TC_LLM_MODEL")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--cpu-url", default=CPU_URL)
    parser.add_argument("--igpu-url", default=IGPU_URL)
    parser.add_argument("--json", type=Path, default=None, help="grava o bruto")
    args = parser.parse_args()

    model = args.model or get_settings().llm_model
    print(f"modelo: {model}   repetições: {args.reps}   geração: {GEN_TOKENS} tokens")

    results: dict[str, dict] = {}

    if not alive(args.cpu_url):
        print(f"\nServidor da CPU fora em {args.cpu_url}. Suba `ollama serve`.")
        return 1
    results["cpu"] = measure(args.cpu_url, model, args.reps)
    report(f"CPU pura — {args.cpu_url}", results["cpu"])

    if not alive(args.igpu_url):
        print(
            f"\nServidor da iGPU fora em {args.igpu_url}. Para subir:\n"
            "  OLLAMA_HOST=127.0.0.1:11435 OLLAMA_VULKAN=1 OLLAMA_IGPU_ENABLE=1 \\\n"
            "      ollama serve"
        )
        return 1
    results["igpu"] = measure(args.igpu_url, model, args.reps)
    report(f"iGPU Vulkan — {args.igpu_url}", results["igpu"])

    cpu, igpu = results["cpu"], results["igpu"]
    print("\n### iGPU contra CPU")
    before, after = cpu["gen_tok_s"], igpu["gen_tok_s"]
    ratio = after / before if before else 0.0
    verdict = "ganho" if ratio > 1.1 else ("empate" if ratio > 0.9 else "perda")
    print(f"  geração        {before:7.1f} → {after:7.1f} tok/s   {ratio:.2f}×  {verdict}")
    for size in PROMPT_SIZES:
        before = cpu["prompt_curve"][size]["tok_s"]
        after = igpu["prompt_curve"][size]["tok_s"]
        ratio = after / before if before else 0.0
        verdict = "ganho" if ratio > 1.1 else ("empate" if ratio > 0.9 else "perda")
        print(f"  prompt ~{size:<4}   {before:7.1f} → {after:7.1f} tok/s   {ratio:.2f}×  {verdict}")

    if igpu["size_vram_gb"] <= 0:
        print("\n  ATENÇÃO: size_vram=0 — o servidor da iGPU não offloadou nada.")
    if igpu["gpu_busy_pct"] < 5:
        print("\n  ATENÇÃO: motor de render quase parado — a GPU não está no caminho.")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nbruto em {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
