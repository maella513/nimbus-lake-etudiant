"""Bloc 1, partie B — reprendre le script hérité en pipeline Bronze -> Silver -> Gold.

Bronze : les fichiers tels que livrés (data/bronze/{ref,mf,static}).
Silver : une table canonique `observations`, une ligne par (ic_id, ts), unités SI
         usuelles (°C, mm), horodatage UTC, doublons retirés, valeurs NON filtrées.
Gold   : la climatologie journalière `journaliere`, contrôle qualité appliqué.

Critères de réussite (tests/test_b1_pipeline.py) :
1. tn et tx identiques au script hérité, à 1e-9 près (il est juste sur ce point) ;
2. rr et n_obs conformes à la spécification ci-dessous (le script hérité est faux) ;
3. le pipeline travaille mois par mois : jamais plus d'un mois de Bronze en mémoire.

Spécification Gold (inspirée de gold_v2.journaliere sur portail.chom.engineering) :
- date      : date UTC de `ts` ;
- n_obs     : nombre d'observations de température non nulles du jour (après dédoublonnage) ;
- n_obs_rejetees : parmi elles, celles hors des bornes physiques BORNES_T ;
- tn, tx    : min et max des températures retenues ; tm = (tn + tx) / 2 ;
- rr        : somme des cumuls horaires `rr1_mm` du jour, null s'il n'y en a aucun ;
- statut_extremes, selon r = n_obs - n_obs_rejetees : "quotidien" si r >= 8,
  "partiel" si 2 <= r <= 7, "releve_unique" si r == 1, null si r == 0.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.json as pajson
import pyarrow.parquet as pq

from nimbus_lake.b1_parquet import lire_bronze_mf
from nimbus_lake.donnees import BORNES_T, SCHEMA_SILVER

SCHEMA_REF = pa.schema([
    pa.field("ic_id", pa.string()),
    pa.field("mfid", pa.string()),
    pa.field("genre", pa.string()),
    pa.field("libelle_source", pa.string()),
    pa.field("latitude", pa.float64()),
    pa.field("longitude", pa.float64()),
    pa.field("altitude", pa.int32()),
    pa.field("departement", pa.string()),
    pa.field("dh_ouverture", pa.timestamp("s")),
])

SCHEMA_GOLD = pa.schema([
    pa.field("ic_id", pa.string(), nullable=False),
    pa.field("mfid", pa.string()),
    pa.field("date", pa.date32(), nullable=False),
    pa.field("tn", pa.float64()),
    pa.field("tx", pa.float64()),
    pa.field("tm", pa.float64()),
    pa.field("rr", pa.float64()),
    pa.field("n_obs", pa.int64(), nullable=False),
    pa.field("n_obs_rejetees", pa.int64(), nullable=False),
    pa.field("statut_extremes", pa.string()),
])

SCHEMA_STATIC_BRUT = pa.schema([
    pa.field("ic_id", pa.string()),
    pa.field("dh_utc", pa.string()),
    pa.field("air_temperature", pa.float64()),
    pa.field("relative_humidity", pa.float64()),
    pa.field("precip_1h", pa.float64()),
])

_MOIS = re.compile(r"^(\d{6})(?:_r\d+)?$")


def lire_referentiel(chemin: Path) -> pa.Table:
    """B1. Lit bronze/ref/stations.csv avec SCHEMA_REF.

    - mfid vide -> null ;
    - dh_ouverture « 0000-00-00 00:00:00 » (date inconnue) -> null, pas une erreur de parsing.
    """
    return pacsv.read_csv(
        chemin,
        convert_options=pacsv.ConvertOptions(
            column_types=SCHEMA_REF,
            include_columns=SCHEMA_REF.names,
            null_values=["", "0000-00-00 00:00:00"],
            strings_can_be_null=True,
        ),
    )


def canoniser_mf(brut: pa.Table, crosswalk: Mapping[str, str]) -> pa.Table:
    """B2. Table Bronze Météo-France (SCHEMA_MF_BRUT) -> SCHEMA_SILVER.

    - ic_id : via le crosswalk NUM_POSTE -> ic_id du référentiel ;
    - ts    : AAAAMMJJHH en UTC ;
    - t_c = T, u_pct = U, rr1_mm = RR1, source = "mf".
    """
    postes = pa.array(list(crosswalk.keys()), pa.string())
    ic_ids = pa.array(list(crosswalk.values()), pa.string())
    ic_id = pc.take(ic_ids, pc.index_in(brut["NUM_POSTE"], value_set=postes))
    iso = pc.replace_substring_regex(
        brut["AAAAMMJJHH"],
        pattern=r"^(\d{4})(\d{2})(\d{2})(\d{2})$",
        replacement=r"\1-\2-\3 \4:00:00",
    )
    ts = iso.cast(pa.timestamp("us")).cast(pa.timestamp("us", tz="UTC"))
    return pa.table({
        "ic_id": ic_id,
        "ts": ts,
        "t_c": brut["T"],
        "u_pct": brut["U"],
        "rr1_mm": brut["RR1"],
        "source": pa.array(["mf"] * brut.num_rows, pa.string()),
    }).cast(SCHEMA_SILVER)

def lire_bronze_static(chemin: Path) -> pa.Table:
    """B3. Fichier JSON Lines du réseau participatif -> SCHEMA_SILVER.

    - dh_utc s'écrit « …Z » ou « …+00:00 » : même instant ;
    - air_temperature est en kelvin : t_c = air_temperature - 273.15 ;
    - precip_1h est un cumul GLISSANT sur 60 min publié toutes les 15 min.
      Seul le relevé de la minute 00 est un cumul horaire : rr1_mm = precip_1h à
      la minute 00, null aux minutes 15, 30 et 45 ;
    - source = "static".
    """
    brut = pajson.read_json(
        chemin,
        parse_options=pajson.ParseOptions(
            explicit_schema=SCHEMA_STATIC_BRUT,
            unexpected_field_behavior="ignore",
        ),
    )
    ts = brut["dh_utc"].cast(pa.timestamp("us", tz="UTC"))
    rr = pc.if_else(pc.equal(pc.minute(ts), 0), brut["precip_1h"],
                    pa.scalar(None, pa.float64()))
    return pa.table({
        "ic_id": brut["ic_id"],
        "ts": ts,
        "t_c": pc.subtract(brut["air_temperature"], 273.15),
        "u_pct": brut["relative_humidity"],
        "rr1_mm": rr,
        "source": pa.array(["static"] * brut.num_rows, pa.string()),
    }).cast(SCHEMA_SILVER)


def mois_disponibles(racine_bronze: Path) -> list[str]:
    """Fourni. Mois (AAAAMM) présents dans bronze/mf ou bronze/static, triés."""
    mois: set[str] = set()
    for chemin in [*(racine_bronze / "mf").glob("*.csv"), *(racine_bronze / "static").glob("*.jsonl")]:
        m = _MOIS.match(chemin.stem)
        if m:
            mois.add(m.group(1))
    return sorted(mois)

def _dedoublonner(table: pa.Table) -> pa.Table:
    """Garde la dernière ligne de chaque (ic_id, ts), puis trie par (ic_id, ts)."""
    indexee = table.append_column("_i", pa.array(range(table.num_rows), pa.int64()))
    derniers = indexee.group_by(["ic_id", "ts"]).aggregate([("_i", "max")])
    return table.take(derniers["_i_max"]).sort_by(
        [("ic_id", "ascending"), ("ts", "ascending")]
    )


def silver_par_mois(racine_bronze: Path) -> Iterator[pa.Table]:
    """B4. Générateur : une table Silver par mois, dans l'ordre chronologique.

    Pour chaque mois : lire TOUS les fichiers du mois (dont les rattrapages
    AAAAMM_r1.csv) avec `lire_bronze_mf` et `lire_bronze_static`, canoniser,
    concaténer, retirer les doublons (même ic_id et même ts), trier par (ic_id, ts).
    Ne lisez pas le mois suivant avant d'avoir rendu le mois courant (`yield`).
    """
    ref = lire_referentiel(racine_bronze / "ref" / "stations.csv")
    crosswalk = {m: i for m, i in zip(ref["mfid"].to_pylist(), ref["ic_id"].to_pylist()) if m}
    for mois in mois_disponibles(racine_bronze):
        tables = [
            canoniser_mf(lire_bronze_mf(c), crosswalk)
            for c in sorted((racine_bronze / "mf").glob(f"{mois}*.csv"))
        ]
        tables += [
            lire_bronze_static(c)
            for c in sorted((racine_bronze / "static").glob(f"{mois}*.jsonl"))
        ]
        yield _dedoublonner(pa.concat_tables(tables))


def gold_journalier(silver: pa.Table, referentiel: pa.Table) -> pa.Table:
    """B5. Silver -> Gold journalier (SCHEMA_GOLD), trié par (date, ic_id).

    `referentiel` (SCHEMA_REF) fournit le mfid de chaque station.
    """
    bas, haut = BORNES_T
    t = silver["t_c"]
    presente = pc.is_valid(t)
    dans_bornes = pc.fill_null(
        pc.and_(pc.greater_equal(t, bas), pc.less_equal(t, haut)), False
    )
    jour = pa.table({
        "ic_id": silver["ic_id"],
        "date": silver["ts"].cast(pa.date32()),
        "presente": presente.cast(pa.int64()),
        "rejetee": pc.and_(presente, pc.invert(dans_bornes)).cast(pa.int64()),
        "t_ok": pc.if_else(dans_bornes, t, pa.scalar(None, pa.float64())),
        "rr": silver["rr1_mm"],
    })
    agg = jour.group_by(["date", "ic_id"]).aggregate([
        ("presente", "sum"), ("rejetee", "sum"),
        ("t_ok", "min"), ("t_ok", "max"), ("rr", "sum"),
    ])
    n_obs = agg["presente_sum"]
    n_rej = agg["rejetee_sum"]
    retenues = pc.subtract(n_obs, n_rej)
    nul = pa.scalar(None, pa.string())
    statut = pc.if_else(
        pc.greater_equal(retenues, 8), pa.scalar("quotidien"),
        pc.if_else(
            pc.greater_equal(retenues, 2), pa.scalar("partiel"),
            pc.if_else(pc.equal(retenues, 1), pa.scalar("releve_unique"), nul),
        ),
    )
    tn, tx = agg["t_ok_min"], agg["t_ok_max"]
    mfid = pc.take(
        referentiel["mfid"].combine_chunks(),
        pc.index_in(agg["ic_id"], value_set=referentiel["ic_id"].combine_chunks()),
    )
    gold = pa.table({
        "ic_id": agg["ic_id"],
        "mfid": mfid,
        "date": agg["date"],
        "tn": tn,
        "tx": tx,
        "tm": pc.divide(pc.add(tn, tx), 2.0),
        "rr": agg["rr_sum"],
        "n_obs": n_obs,
        "n_obs_rejetees": n_rej,
        "statut_extremes": statut,
    })
    return gold.sort_by([("date", "ascending"), ("ic_id", "ascending")]).cast(SCHEMA_GOLD)

@dataclass(frozen=True, slots=True)
class RapportPipeline:
    n_mois: int
    n_lignes_silver: int
    n_lignes_gold: int
    fichiers_silver: tuple[Path, ...]
    fichier_gold: Path


def executer_pipeline(racine_bronze: Path, racine_lake: Path) -> RapportPipeline:
    """B6. Coquille impérative : écrit le Silver et le Gold.

    - Silver : racine_lake/silver/observations/annee=AAAA/AAAAMM.parquet (zstd) ;
    - Gold   : racine_lake/gold/journaliere.parquet, écrit mois par mois avec un
      pq.ParquetWriter (un row group par mois) ;
    - mémoire : un seul mois à la fois (consommer `silver_par_mois` en flux).
    """
    ref = lire_referentiel(racine_bronze / "ref" / "stations.csv")
    fichier_gold = racine_lake / "gold" / "journaliere.parquet"
    fichier_gold.parent.mkdir(parents=True, exist_ok=True)
    fichiers_silver: list[Path] = []
    n_silver = 0
    n_gold = 0
    with pq.ParquetWriter(fichier_gold, SCHEMA_GOLD, compression="zstd") as ecrivain:
        for mois, silver in zip(mois_disponibles(racine_bronze), silver_par_mois(racine_bronze)):
            dossier = racine_lake / "silver" / "observations" / f"annee={mois[:4]}"
            dossier.mkdir(parents=True, exist_ok=True)
            chemin = dossier / f"{mois}.parquet"
            pq.write_table(silver, chemin, compression="zstd")
            fichiers_silver.append(chemin)
            n_silver += silver.num_rows
            gold = gold_journalier(silver, ref)
            ecrivain.write_table(gold)
            n_gold += gold.num_rows
    return RapportPipeline(
        n_mois=len(fichiers_silver),
        n_lignes_silver=n_silver,
        n_lignes_gold=n_gold,
        fichiers_silver=tuple(fichiers_silver),
        fichier_gold=fichier_gold,
    )
