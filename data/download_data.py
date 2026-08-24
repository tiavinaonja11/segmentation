"""
Télécharge et organise le dataset public AI4Boundaries (Sentinel-2) publié par le
Joint Research Centre (JRC) de la Commission Européenne.

Référence :
    d'Andrimont, R., Claverie, M., Kempeneers, P., Muraro, D., Yordanov, M.,
    Peressutti, D., Batič, M., and Waldner, F. (2023): "AI4Boundaries: an open
    AI-ready dataset to map field boundaries with Sentinel-2 and aerial
    photography", Earth Syst. Sci. Data, 15, 317-329.
    https://doi.org/10.5194/essd-15-317-2023

Source des données (accès FTP libre, sans authentification) :
    https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/

⚠️ IMPORTANT : ce script doit être exécuté sur VOTRE machine (avec accès internet
complet). Le sandbox utilisé pour générer ce projet n'a pas accès à ce serveur FTP.

Structure téléchargée (extrait Sentinel-2, images + masques déjà prêts à l'emploi) :
    sentinel2/images/<country>/*.tif   (10 bandes, composite mensuel)
    sentinel2/masks/<country>/*.tif    (masque binaire de parcelle + masque de bordure)

Usage:
    python data/download_data.py --country FR --out data/ai4boundaries
"""

import os
import argparse
import urllib.request
import zipfile

BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/sentinel2"

COUNTRY_CODES = {
    "AT": "Autriche",
    "ES": "Catalogne (Espagne)",
    "FR": "France",
    "LU": "Luxembourg",
    "NL": "Pays-Bas",
    "SI": "Slovénie",
    "SE": "Suède",
}


def download_file(url, dest_path):
    print(f"Téléchargement : {url}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"  -> sauvegardé dans {dest_path}")


def download_country(country_code, out_dir):
    if country_code not in COUNTRY_CODES:
        raise ValueError(
            f"Code pays inconnu '{country_code}'. Choix possibles : {list(COUNTRY_CODES.keys())}"
        )

    # Le nom exact des archives peut évoluer sur le serveur JRC : vérifiez la
    # liste des fichiers disponibles sur la page FTP avant de lancer ce script :
    # https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/sentinel2/
    zip_name = f"{country_code}.zip"
    url = f"{BASE_URL}/{zip_name}"
    dest_zip = os.path.join(out_dir, zip_name)

    download_file(url, dest_zip)

    print("Extraction de l'archive...")
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(out_dir)

    print(f"\n✅ Données AI4Boundaries pour '{COUNTRY_CODES[country_code]}' prêtes dans {out_dir}")
    print(
        "\n👉 Étape suivante : réorganisez/convertissez si besoin les fichiers en "
        "sous-dossiers 'images/' et 'masks/' attendus par data/dataset.py, "
        "puis lancez train.py."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharge le dataset AI4Boundaries (JRC)")
    parser.add_argument(
        "--country", type=str, default="FR",
        help=f"Code pays parmi {list(COUNTRY_CODES.keys())}",
    )
    parser.add_argument("--out", type=str, default="data/ai4boundaries", help="Dossier de sortie")
    args = parser.parse_args()

    print("=" * 70)
    print("Téléchargement du dataset AI4Boundaries (JRC, Commission Européenne)")
    print("=" * 70)
    print(
        "\nSi ce script échoue (structure de fichiers changée sur le serveur), "
        "téléchargez manuellement depuis :"
    )
    print("  https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/")
    print("ou depuis le catalogue JRC officiel :")
    print("  http://data.europa.eu/89h/0e79ce5d-e4c8-4721-8773-59a4acf2c9c9\n")

    try:
        download_country(args.country, args.out)
    except Exception as e:
        print(f"\n❌ Échec du téléchargement automatique : {e}")
        print("Téléchargez manuellement via les liens ci-dessus, puis placez les")
        print("fichiers .tif dans data/ai4boundaries/images/ et data/ai4boundaries/masks/")
