"""Bloc 1, partie A — Parquet de l'intérieur.

Un fichier Parquet est découpé en row groups ; chaque row group stocke chaque
colonne séparément (column chunk), encodée puis compressée. Le footer, à la fin du
fichier, contient le schéma et, pour chaque column chunk, des statistiques min/max.
Un moteur lit le footer d'abord et saute les row groups dont les statistiques
excluent le prédicat : c'est le predicate pushdown au niveau fichier.

Exercices A1 à A7. Tests : `uv run pytest tests/test_b1_parquet.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# Schéma Bronze du CSV Météo-France. NUM_POSTE est un identifiant, pas un nombre :
# « 01918007 » doit garder son zéro initial. AAAAMMJJHH reste un texte au Bronze.
SCHEMA_MF_BRUT = pa.schema([
    pa.field("NUM_POSTE", pa.string()),
    pa.field("AAAAMMJJHH", pa.string()),
    pa.field("T", pa.float64()),
    pa.field("U", pa.float64()),
    pa.field("PMER", pa.float64()),
    pa.field("RR1", pa.float64()),
])


def lire_bronze_mf(chemin: Path) -> pa.Table:
    """A1. Lit un CSV Météo-France (séparateur « ; ») avec le schéma SCHEMA_MF_BRUT.

    Ne laissez pas pyarrow inférer les types : il ferait de NUM_POSTE un entier.
    Une cellule vide devient null.
    """
    return pacsv.read_csv(
        chemin,
        parse_options=pacsv.ParseOptions(delimiter=";"),
        convert_options=pacsv.ConvertOptions(
            column_types=SCHEMA_MF_BRUT,
            include_columns=SCHEMA_MF_BRUT.names,
        ),
    )


def ecrire_parquet(table: pa.Table, chemin: Path, compression: str | None,
                   lignes_par_row_group: int) -> int:
    """A2. Écrit `table` en Parquet et renvoie la taille du fichier en octets.

    `compression` : None, "snappy" ou "zstd".
    """
    pq.write_table(
        table,
        chemin,
        compression=compression if compression is not None else "none",
        row_group_size=lignes_par_row_group,
    )
    return chemin.stat().st_size


@dataclass(frozen=True, slots=True)
class InfosParquet:
    n_lignes: int
    n_row_groups: int
    lignes_par_row_group: tuple[int, ...]
    colonnes: tuple[str, ...]
    codecs: frozenset[str]          # ex. frozenset({"ZSTD"})
    taille_footer: int              # octets du footer sérialisé (FileMetaData)


def inspecter(chemin: Path) -> InfosParquet:
    """A3. Lit uniquement le footer (pq.ParquetFile(...).metadata) et le résume."""
    meta = pq.ParquetFile(chemin).metadata
    n_groupes = meta.num_row_groups
    return InfosParquet(
        n_lignes=meta.num_rows,
        n_row_groups=n_groupes,
        lignes_par_row_group=tuple(meta.row_group(i).num_rows for i in range(n_groupes)),
        colonnes=tuple(meta.schema.names),
        codecs=frozenset(
            meta.row_group(i).column(j).compression
            for i in range(n_groupes)
            for j in range(meta.num_columns)
        ),
        taille_footer=meta.serialized_size,
    )


def bornes_row_groups(chemin: Path, colonne: str) -> list[tuple[object, object]]:
    """A4. (min, max) de `colonne` pour chaque row group, lus dans les statistiques du footer."""
    meta = pq.ParquetFile(chemin).metadata
    j = meta.schema.names.index(colonne)
    bornes: list[tuple[object, object]] = []
    for i in range(meta.num_row_groups):
        stats = meta.row_group(i).column(j).statistics
        if stats is None or not stats.has_min_max:
            bornes.append((None, None))
        else:
            bornes.append((stats.min, stats.max))
    return bornes


def row_groups_candidats(chemin: Path, colonne: str, valeur: object) -> list[int]:
    """A5. Indices des row groups qui PEUVENT contenir `colonne == valeur`.

    C'est le raisonnement d'un moteur avant de lire les données : un row group dont
    l'intervalle [min, max] exclut la valeur est sauté sans être décompressé.
    """
    candidats: list[int] = []
    for i, (bas, haut) in enumerate(bornes_row_groups(chemin, colonne)):
        if bas is None or haut is None:
            candidats.append(i)  # pas de statistiques : impossible d'exclure
        elif cast(Any, bas) <= cast(Any, valeur) <= cast(Any, haut):
            candidats.append(i)
    return candidats


def tailles_par_format(table: pa.Table, dossier: Path) -> dict[str, int]:
    """A6. Écrit `table` en CSV puis en Parquet sans compression, snappy et zstd.

    Renvoie {"csv": ..., "parquet_none": ..., "parquet_snappy": ..., "parquet_zstd": ...}
    en octets. Row groups de 100 000 lignes.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    chemin_csv = dossier / "mf.csv"
    pacsv.write_csv(table, chemin_csv)
    tailles = {"csv": chemin_csv.stat().st_size}
    for nom, codec in (("none", None), ("snappy", "snappy"), ("zstd", "zstd")):
        tailles[f"parquet_{nom}"] = ecrire_parquet(
            table, dossier / f"mf_{nom}.parquet", codec, 100_000
        )
    return tailles

def effet_du_tri(table: pa.Table, colonne: str, valeur: object, dossier: Path,
                 lignes_par_row_group: int) -> tuple[int, int]:
    """A7. Nombre de row groups candidats pour `colonne == valeur`, sans puis avec tri.

    Écrit la table telle quelle, puis triée sur `colonne`, et compare. Renvoie
    (candidats_sans_tri, candidats_avec_tri).
    """
    sans_tri = dossier / "sans_tri.parquet"
    avec_tri = dossier / "avec_tri.parquet"
    ecrire_parquet(table, sans_tri, "zstd", lignes_par_row_group)
    ecrire_parquet(table.sort_by(colonne), avec_tri, "zstd", lignes_par_row_group)
    return (
        len(row_groups_candidats(sans_tri, colonne, valeur)),
        len(row_groups_candidats(avec_tri, colonne, valeur)),
    )