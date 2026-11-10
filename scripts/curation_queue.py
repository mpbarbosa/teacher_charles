#!/usr/bin/env python3
"""A fila de curadoria: os erros que a base contrastiva ainda não cobre.

Lê `unverified_finding` — as correções que o verificador da E4 barrou por não
citarem regra de `data/contrastive/` — agregadas por trecho do aluno, do mais
frequente para baixo. É o material para escrever item novo de YAML, ou para
alargar o padrão de um item que já existe:

    uv run python scripts/curation_queue.py
        A fila, 20 linhas.

    uv run python scripts/curation_queue.py --limit 50
    uv run python scripts/curation_queue.py --yaml
        O esqueleto de `ContrastiveItem` de cada buraco, para colar no YAML
        e editar. Esqueleto, e não item: `pt`, `explanation` e `patterns`
        são trabalho humano — e `explanation` em inglês, decisão 1.1 do plano.

## Por que isto é um script e não tela

Quem lê é quem escreve YAML, não quem está aprendendo inglês. E o que está
aqui dentro é a gramática que o modelo pequeno improvisou e a E4 barrou: vê-la
é útil para curar, e seria nocivo para o aluno — a mesma razão de
`Analysis.as_dict` não expor estes achados ao SSE.

## O que a fila não vê

Com `require_curated_rule` ligado (o default), a análise não roda quando a
base não levanta candidato nenhum. Sem chamada não há achado, e sem achado não
há linha: **toda linha daqui vem de um turno em que a base já acertou alguma
outra regra**. O erro que a base não enxerga de jeito nenhum é invisível por
construção, e não dá para fechar isso com dado — sem rodar o modelo, "inglês
correto" e "erro que a base não vê" produzem a mesma ausência de candidato.

Para ir atrás desse buraco, rode uma sessão com `TC_REQUIRE_CURATED_RULE=false`
e volte aqui: a análise passa a rodar em todo turno e a fila cobre o inverso.
Isso custa a latência que o default protege — é rodada de curadoria
deliberada, não uso normal.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from teacher_charles.config import get_settings  # noqa: E402
from teacher_charles.store.models import GapCount  # noqa: E402
from teacher_charles.store.repo import UnverifiedFindingRepo  # noqa: E402

# O id tem a categoria no prefixo (`calque.age-have`), e a categoria do
# esqueleto sai do `error_type` que o modelo propôs. Quando não dá para mapear,
# o esqueleto vem com o prefixo em branco para o curador decidir -- chutar
# categoria aqui faria o `_id_prefix_matches_category` do item barrar a carga
# com uma mensagem que não ajuda ninguém.
TYPE_TO_CATEGORY = {
    "calque": "calque",
    "false-friend": "false-friend",
    "false_friend": "false-friend",
    "grammar": "grammar",
    "pronunciation": "pronunciation",
    "vocabulary": "false-friend",
    "word-choice": "false-friend",
}


def _category_for(error_type: str) -> str:
    return TYPE_TO_CATEGORY.get(error_type.strip().lower(), "")


def render_table(queue: list[GapCount]) -> str:
    if not queue:
        return (
            "Fila vazia.\n\n"
            "Ou a base cobre tudo o que apareceu, ou nenhum turno chegou a ter\n"
            "análise com achado barrado -- veja o ponto cego no topo deste script."
        )

    lines = [f"{len(queue)} buraco(s), do mais frequente para baixo:", ""]
    for gap in queue:
        lines.append(f"  {gap.times:>3}x  {gap.last_span!r}")
        lines.append(f"        o modelo propôs: {gap.last_correction!r}  [{gap.last_type}]")
        if gap.last_candidates:
            # Casou outra regra no mesmo turno: provável padrão a alargar.
            lines.append(f"        ao lado de:      {gap.last_candidates}")
        else:
            lines.append("        ao lado de:      (nada -- veja o ponto cego)")
        lines.append(f"        último em:       {gap.last_at}")
        lines.append("")
    return "\n".join(lines)


def _yaml_item(value: str) -> str:
    """O trecho do aluno como item de lista YAML válido.

    Via `safe_dump`, e não `repr`: o span é texto do aluno e pode ter apóstrofo
    e aspas na mesma frase (*don't say "hi"*), caso em que o `repr` do Python
    escapa com barra invertida e o YAML resultante não carrega. O esqueleto
    existe para ser colado no arquivo — um que não faz parse não serve.
    """
    return yaml.safe_dump([value], allow_unicode=True, default_flow_style=False).strip()


def render_yaml(queue: list[GapCount]) -> str:
    if not queue:
        return "# Fila vazia."

    lines = [
        "# Esqueletos da fila de curadoria. NÃO são itens prontos:",
        "#   - `pt` é o português que causa o erro, e só quem cura sabe qual é;",
        "#   - `explanation` vai para o aluno, em inglês (decisão 1.1 do plano);",
        "#   - `patterns` é o que faz o item generalizar, e sem ele os literais",
        "#     de `wrong` viram o padrão -- o bastante só para forma fixa;",
        "#   - o id precisa começar pela categoria, senão a carga barra.",
        "# Cada item ganha teste, como todo o resto da base.",
        "items:",
    ]
    for gap in queue:
        category = _category_for(gap.last_type)
        prefix = f"{category}." if category else "TODO."
        lines += [
            f"  # {gap.times}x; o modelo chamou de {gap.last_type!r}",
            f"  #   ele explicou assim: {gap.last_explanation}",
            f"  - id: {prefix}TODO",
            f"    category: {category or 'TODO'}",
            '    pt: "TODO"',
            "    wrong:",
            f"      {_yaml_item(gap.last_span)}",
            "    right:",
            f"      {_yaml_item(gap.last_correction)}",
            "    explanation: >-",
            "      TODO",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--limit", type=int, default=20, help="quantos buracos listar")
    parser.add_argument("--yaml", action="store_true", help="esqueleto de item para colar")
    args = parser.parse_args(argv)

    settings = get_settings()
    db_path = settings.resolved_db_path
    if not db_path.exists():
        print(f"sem banco em {db_path} -- nenhuma aula aconteceu ainda.", file=sys.stderr)
        return 1

    # Somente leitura, e por isso não passa por `init_db`: um script de
    # curadoria não tem por que aplicar migração no banco do aluno.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        repo = UnverifiedFindingRepo(conn)
        try:
            queue = repo.backlog(limit=args.limit)
            total, distinct = repo.count(), repo.distinct_spans()
        except sqlite3.OperationalError:
            print(
                "o banco não tem `unverified_finding` -- ele é anterior à migração 005.\n"
                "Suba o servidor uma vez para migrar.",
                file=sys.stderr,
            )
            return 1
    finally:
        conn.close()

    print(render_yaml(queue) if args.yaml else render_table(queue))
    if queue and not args.yaml:
        print(f"{total} achado(s) barrado(s) no total, {distinct} trecho(s) distinto(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
