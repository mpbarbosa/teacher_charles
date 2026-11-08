#!/usr/bin/env bash
#
# Sobe o Teacher Charles. `scripts/run.sh --help` lista os modos.
#
# Why a wrapper and not an alias: the ways of running this app differ only in
# environment, and the two that matter are easy to get half-right by hand.
# Voice needs three things at once — the `voice` extra, the weights on disk and
# `TC_VOICE_ENABLED=true` — and two of the three fail quietly: without the
# extra the engines fail to load lazily and only `/health` says so, and without
# the weights Kokoro raises on the first sentence, after the student has
# already spoken. The fake provider needs no model at all, which is the one
# mode whose failure is loud and whose setup is trivial to forget.
#
# What it deliberately does NOT do: `uv sync` on every start. `uv sync` prunes
# whatever the requested extras do not name, so syncing with `--extra dev`
# alone uninstalls the ~400 MB voice stack — and with it the measurement
# workflow of docs/plano-fase-2.md, which needs `onnxruntime` and the 318 MB
# phoneme weight. A launcher that quietly undoes someone's environment is
# worse than no launcher, so syncing is opt-in and `--sync --voice` keeps both
# extras.

set -euo pipefail

# From the script's own directory, so it works from anywhere — including from
# a worktree, where a hardcoded path would point at the wrong checkout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
    cat <<'EOF'
Uso: scripts/run.sh [opções]

Modos
  (nenhuma opção)   Ollama local, como o README documenta
  --fake            sem modelo nenhum, para trabalhar na interface
  --voice           liga a voz: extra, pesos e TC_VOICE_ENABLED=true
  --model NOME      usa um modelo já baixado (TC_LLM_MODEL)

Antes de subir
  --check           roda scripts/check_env.py e para se houver bloqueio
  --sync            instala as dependências (com --voice, inclui o extra voice)

Repassadas ao servidor
  --reload          recarrega a cada mudança de arquivo
  --host ENDEREÇO   default 127.0.0.1
  --port PORTA      default 8000

  -h, --help        esta ajuda

Exemplos
  scripts/run.sh --fake --reload           # interface, sem modelo
  scripts/run.sh --check                   # confere o ambiente e sobe
  scripts/run.sh --sync --voice            # primeira vez com voz
EOF
}

die() {
    echo "run.sh: $*" >&2
    exit 1
}

fake=false
voice=false
check=false
sync=false
model=""
# Everything the server parses itself goes through untouched: --host, --port
# and --reload are `teacher-charles`'s own flags, and duplicating their
# defaults here would be two places to keep in step with cli.py.
passthrough=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --fake) fake=true ;;
        --voice) voice=true ;;
        --check) check=true ;;
        --sync) sync=true ;;
        --model)
            [[ $# -ge 2 ]] || die "--model precisa do nome do modelo"
            model="$2"
            shift
            ;;
        --reload) passthrough+=("$1") ;;
        --host | --port)
            [[ $# -ge 2 ]] || die "$1 precisa de um valor"
            passthrough+=("$1" "$2")
            shift
            ;;
        *) die "opção desconhecida: $1 (veja --help)" ;;
    esac
    shift
done

$fake && [[ -n "$model" ]] && die "--fake e --model se contradizem: o fake não carrega modelo"

$fake && export TC_PROVIDER=fake
[[ -n "$model" ]] && export TC_LLM_MODEL="$model"

if $check; then
    echo "==> checagem de ambiente"
    # With the system python, not `uv run`: this is the one check meant to run
    # *before* anything is installed. It imports only the stdlib, and the
    # README sells it as asking the machine whether it is ready without
    # installing anything — through `uv run` it would create and sync the venv
    # first, which is the opposite of that.
    #
    # It also runs before the `uv` guard below, so a machine without uv gets
    # the whole report instead of one line: missing uv is one of the things it
    # reports. And after the exports above, because it reads TC_LLM_MODEL — with
    # `--model` it should judge the model this run is about to use.
    if python3 scripts/check_env.py; then
        :
    elif $fake; then
        # `check_env.py` does not read TC_PROVIDER: it always probes Ollama and
        # the configured model. Under --fake there is no model to probe, so its
        # blocker is about something this run does not touch — and aborting on
        # it would refuse the one mode that works on a machine with nothing
        # installed, which is the mode someone reaches for first.
        echo "run.sh: a checagem achou bloqueio, mas --fake não usa modelo; seguindo" >&2
    else
        die "a checagem achou bloqueio; nada foi iniciado"
    fi
fi

command -v uv >/dev/null 2>&1 ||
    die "uv não está no PATH. Instale com: curl -LsSf https://astral.sh/uv/install.sh | sh"

if $sync; then
    extras=(--extra dev)
    $voice && extras+=(--extra voice)
    echo "==> uv sync ${extras[*]}"
    uv sync "${extras[@]}"
fi

if $voice; then
    # Idempotent by design — it prints "já está lá" for a weight it finds — so
    # this costs a second on every voiced start and removes the failure that
    # only shows up when the student has already spoken.
    echo "==> pesos da voz"
    uv run python scripts/fetch_voice_models.py

    # The extra is the half that `fetch_voice_models.py` cannot check: it runs
    # on the base dependencies, so it succeeds with the engines absent. Without
    # this guard the server comes up, answers, and stays mute — with the reason
    # only in `/health`.
    uv run python -c 'import faster_whisper, kokoro_onnx' 2>/dev/null ||
        die "a voz pede o extra: rode outra vez com --sync --voice"

    export TC_VOICE_ENABLED=true
fi

# `exec`, so Ctrl-C reaches uvicorn instead of this shell, and so the server is
# the process the terminal is waiting on.
exec uv run teacher-charles "${passthrough[@]}"
