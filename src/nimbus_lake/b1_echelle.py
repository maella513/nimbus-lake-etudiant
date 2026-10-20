"""Bloc 1, partie C — partitioning à l'échelle (200 à 1 000 stations × 10 ans).

Le partitioning « hive » range les fichiers dans des dossiers `cle=valeur/`. Un
moteur qui reçoit un filtre sur la clé ne liste même pas les autres dossiers :
c'est le partition pruning, qui agit AVANT le predicate pushdown sur les row groups
de la partie A. Mal choisie, la clé produit des milliers de petits fichiers
(small files problem) : chaque fichier coûte une ouverture, un footer, une requête
S3 dans le cloud.

Exercices C1 à C5, bonus. Tests : `uv run pytest tests/test_b1_echelle.py`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from nimbus_lake.donnees import SCHEMA_SILVER

CLES: dict[str, tuple[str, ...]] = {
    "aucune": (),
    "annee": ("annee",),
    "annee_mois": ("annee", "mois"),
    "ic_id": ("ic_id",),
    "ic_id_annee": ("ic_id", "annee"),
}

SCHEMA_PARTITIONNABLE = SCHEMA_SILVER.append(pa.field("annee", pa.int16())).append(
    pa.field("mois", pa.int8()))


def compter_fichiers(racine: Path) -> int:
    """Fourni. Nombre de fichiers Parquet sous `racine`."""
    return sum(1 for _ in racine.rglob("*.parquet"))


def ajouter_colonnes_partition(table: pa.Table) -> pa.Table:
    """C1. Ajoute les colonnes `annee` (int16) et `mois` (int8), tirées de `ts` (UTC).

    Le résultat suit SCHEMA_PARTITIONNABLE.
    """
    annee = pc.year(table["ts"]).cast(pa.int16())
    mois = pc.month(table["ts"]).cast(pa.int8())
    return (
        table.append_column("annee", annee)
        .append_column("mois", mois)
        .cast(SCHEMA_PARTITIONNABLE)
    )


def ecrire_partitionne(tables: Iterable[pa.Table], racine: Path, cle: str) -> int:
    """C2. Écrit les tables Silver sous `racine` selon la clé `cle` (voir CLES).

    - partitioning hive (`annee=2020/mois=3/…`) ; "aucune" = pas de dossier ;
    - en flux : ne concaténez pas les tables, passez un itérateur de RecordBatch à
      ds.write_dataset (avec `schema=`) ;
    - renvoie le nombre de fichiers Parquet écrits.
    """
    champs = [SCHEMA_PARTITIONNABLE.field(nom) for nom in CLES[cle]]
    partitionnement = ds.partitioning(pa.schema(champs), flavor="hive") if champs else None

    def lots() -> Iterator[pa.RecordBatch]:
        for table in tables:
            yield from ajouter_colonnes_partition(table).to_batches()

    ds.write_dataset(
        lots(),
        racine,
        schema=SCHEMA_PARTITIONNABLE,
        format="parquet",
        partitioning=partitionnement,
    )
    return compter_fichiers(racine)


def ouvrir(racine: Path) -> ds.Dataset:
    """Fourni. Ouvre un dataset écrit par `ecrire_partitionne`, quelle que soit la clé."""
    return ds.dataset(racine, format="parquet", partitioning="hive")


def fichiers_lus(racine: Path, filtre: pc.Expression) -> int:
    """C3. Nombre de fichiers que le moteur doit ouvrir pour `filtre`, après partition pruning.

    Indice : Dataset.get_fragments(filter=...) n'utilise que les valeurs de partition.
    """
    return len(list(ouvrir(racine).get_fragments(filter=filtre)))


def moyenne_station_annee(racine: Path, ic_id: str, annee: int) -> tuple[int, float]:
    """C4. (nombre d'observations, moyenne de t_c) pour une station et une année.

    Doit fonctionner pour toutes les clés de CLES. Ne chargez que la colonne t_c.
    """
    filtre = (pc.field("ic_id") == ic_id) & (pc.field("annee") == annee)
    t_c = ouvrir(racine).to_table(columns=["t_c"], filter=filtre)["t_c"]
    return len(t_c), float(pc.mean(t_c).as_py())


def compacter(source: Path, cible: Path, cle: str, lignes_par_fichier: int) -> int:
    """C5. Réécrit le dataset `source` sous `cible` avec la clé `cle`.

    - au plus `lignes_par_fichier` lignes par fichier (max_rows_per_file) ;
    - row groups de 128 000 lignes au plus ;
    - renvoie le nombre de fichiers écrits.
    """
    source_ds = ouvrir(source)
    champs = [source_ds.schema.field(nom) for nom in CLES[cle]]
    partitionnement = ds.partitioning(pa.schema(champs), flavor="hive") if champs else None
    ds.write_dataset(
        source_ds,
        cible,
        format="parquet",
        partitioning=partitionnement,
        max_rows_per_file=lignes_par_fichier,
        max_rows_per_group=min(128_000, lignes_par_fichier),
    )
    return compter_fichiers(cible)


def gold_mensuel_duckdb(racine: Path) -> pa.Table:
    """Bonus. Moyenne mensuelle de t_c par station, calculée par DuckDB sur le dataset.

    Colonnes : ic_id (string), annee (int), mois (int), t_moy (double), n (int),
    triées par ic_id, annee, mois. Utilisez read_parquet(..., hive_partitioning = true).
    Comparez ensuite `EXPLAIN ANALYZE` avec et sans filtre sur annee.
    """
    motif = f"{racine.as_posix()}/**/*.parquet"
    return duckdb.sql(
        f"""
        SELECT ic_id, annee, mois, avg(t_c) AS t_moy, count(*) AS n
        FROM read_parquet('{motif}', hive_partitioning = true)
        GROUP BY ic_id, annee, mois
        ORDER BY ic_id, annee, mois
        """
    ).to_arrow_table()


def main() -> None:  # pragma: no cover - coquille impérative
    """Fourni. Banc d'essai de la partie C.

        uv run python -m nimbus_lake.b1_echelle 200          # 200 stations × 10 ans
        uv run python -m nimbus_lake.b1_echelle 1000         # 29 millions de lignes
    """
    import sys
    import time

    from nimbus_lake.donnees import silver_massif

    n_stations = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    racine = Path("data") / f"echelle_{n_stations}"
    print(f"{'clé':<12} {'fichiers':>8} {'Mo':>8} {'écriture s':>11} {'requête ms':>11} {'fichiers lus':>13}")
    for cle in CLES:
        cible = racine / cle
        if cible.exists() and any(cible.iterdir()):
            sys.exit(f"{cible} existe déjà : supprimez data/echelle_{n_stations} avant de relancer.")
        debut = time.perf_counter()
        n_fichiers = ecrire_partitionne(silver_massif(n_stations=n_stations), cible, cle)
        ecriture = time.perf_counter() - debut
        taille = sum(p.stat().st_size for p in cible.rglob("*.parquet")) / 1e6
        debut = time.perf_counter()
        moyenne_station_annee(cible, "ST0042", 2020)
        requete = (time.perf_counter() - debut) * 1000
        lus = fichiers_lus(cible, (pc.field("ic_id") == "ST0042") & (pc.field("annee") == 2020))
        print(f"{cle:<12} {n_fichiers:>8} {taille:>8.1f} {ecriture:>11.1f} {requete:>11.1f} {lus:>13}")


if __name__ == "__main__":  # pragma: no cover
    main()
