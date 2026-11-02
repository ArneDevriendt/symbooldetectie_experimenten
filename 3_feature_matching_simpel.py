"""Programma 3 - Feature matching (SIFT), SIMPELE versie.

Idee
    Een symbool wordt niet pixel per pixel vergeleken, maar via "kenmerken" (keypoints): herkenbare punten
    zoals hoeken en kruisingen, elk met een beschrijving (descriptor) die ongevoelig is voor schaal en rotatie.
    Als genoeg kenmerken van het symbool op een consistente manier in het schema terugkomen, is het symbool gevonden.

Stappen (per symbool)
    1. SIFT (standaardinstellingen) berekent keypoints + descriptors voor het symbool en voor het schema.
    2. Voor elk keypoint van het symbool zoeken we de CANDIDATEN beste overeenkomsten in het schema
       (BFMatcher, k=4). We behouden alle kandidaten die hoogstens AFSTANDSTOLERANTIE (30%) slechter zijn dan
       de beste. (Lowe's ratio-test werkt NIET als een symbool meerdere keren voorkomt: de beste en de
       tweede beste match zijn dan even goed, waardoor ze allebei verworpen worden.)
    3. RANSAC (cv2.estimateAffinePartial2D) zoekt de grootste groep matches die samen bij één plaatsing past
       (verschuiving + schaal + rotatie). Dat is één exemplaar van het symbool.
    4. De omtrek van het symbool wordt met die transformatie in het schema geprojecteerd -> box.
    5. De gebruikte keypoints van het schema worden uitgesloten en we zoeken het volgende exemplaar,
       tot er niet genoeg matches meer zijn (maximaal MAX_EXEMPLAREN per symbool).

Symbolen met te weinig keypoints (< --min-matches) worden overgeslagen; het programma toont het aantal.

Gebruik
    python 3_feature_matching_simpel.py schemas/schema1.png --symbolen symbolen_set1
    python 3_feature_matching_simpel.py schema_rijk.png --symbolen symbolen_set2 --min-matches 8

Uitvoer (in de map --uit, standaard "resultaten"): <programma>__<schema>.png en .txt
    txt: per regel label  x  y  breedte  hoogte  score   (score = inliers / aantal keypoints van het symbool, max 1)
"""
import argparse
import colorsys
import os
import re
import time
import zlib

import cv2
import numpy as np

AFBEELDING_EXTENSIES = (".png", ".jpg", ".jpeg", ".bmp")
CANDIDATEN = 4               # kandidaat-matches per keypoint van het symbool
AFSTANDSTOLERANTIE = 0.30    # kandidaten tot 30% slechter dan de beste blijven over
MAX_EXEMPLAREN = 50          # maximum aantal exemplaren per symbool
RANSAC_DREMPEL = 5.0         # pixels: hoeveel een match mag afwijken van de transformatie


def laad_symbolen(symbolen_map):
    """Geeft een lijst (label, grijswaardenafbeelding)."""
    if not os.path.isdir(symbolen_map):
        raise SystemExit(f"Symbolenmap niet gevonden: {symbolen_map}")
    symbolen = []
    for bestandsnaam in sorted(os.listdir(symbolen_map)):
        if bestandsnaam.lower().endswith(AFBEELDING_EXTENSIES):
            grijs = cv2.imread(os.path.join(symbolen_map, bestandsnaam), cv2.IMREAD_GRAYSCALE)
            if grijs is not None:
                label = re.sub(r"[_-]?\d+$", "", os.path.splitext(bestandsnaam)[0])  # Klep_2 -> Klep
                symbolen.append((label, grijs))
    if not symbolen:
        raise SystemExit(f"Geen afbeeldingen in {symbolen_map}")
    return symbolen


def kleur_van_label(label):
    tint = (zlib.crc32(label.encode()) % 360) / 360.0
    rood, groen, blauw = colorsys.hsv_to_rgb(tint, 0.9, 0.9)
    return int(blauw * 255), int(groen * 255), int(rood * 255)


def teken_detecties(afbeelding, detecties):
    resultaat = afbeelding.copy()
    for detectie in detecties:
        x, y, breedte, hoogte = (int(waarde) for waarde in detectie["box"])
        kleur = kleur_van_label(detectie["label"])
        cv2.rectangle(resultaat, (x, y), (x + breedte, y + hoogte), kleur, 2)
        cv2.putText(resultaat, f'{detectie["label"]} {detectie["score"]:.2f}', (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, kleur, 1, cv2.LINE_AA)
    return resultaat


def sla_resultaat_op(programma_naam, schema_pad, schema, detecties, uit_map, seconden):
    os.makedirs(uit_map, exist_ok=True)
    basis = os.path.join(uit_map, f"{programma_naam}__{os.path.splitext(os.path.basename(schema_pad))[0]}")
    cv2.imwrite(basis + ".png", teken_detecties(schema, detecties))
    with open(basis + ".txt", "w", encoding="utf-8") as bestand:
        for detectie in detecties:
            x, y, breedte, hoogte = (int(waarde) for waarde in detectie["box"])
            bestand.write(f'{detectie["label"]}\t{x}\t{y}\t{breedte}\t{hoogte}\t{detectie["score"]:.4f}\n')
    print(f"{len(detecties)} detectie(s) in {seconden:.1f} s -> {basis}.png en .txt")


def zoek_exemplaren(matcher, symbool_keypoints, symbool_descriptors, symbool_vorm,
                    schema_keypoints, schema_descriptors, min_matches):
    """Alle exemplaren van een symbool: lijst van ((x, y, breedte, hoogte), score)."""
    beschikbaar = np.ones(len(schema_keypoints), dtype=bool)       # nog niet gebruikte keypoints van het schema
    exemplaren = []
    for _ in range(MAX_EXEMPLAREN):
        beschikbare_indices = np.flatnonzero(beschikbaar)
        if len(beschikbare_indices) < min_matches:
            break
        kandidaat_lijsten = matcher.knnMatch(symbool_descriptors, schema_descriptors[beschikbare_indices],
                                             k=min(CANDIDATEN, len(beschikbare_indices)))
        bronpunten, doelpunten, schema_indices = [], [], []
        for kandidaten in kandidaat_lijsten:
            for match in kandidaten:
                if match.distance <= kandidaten[0].distance * (1 + AFSTANDSTOLERANTIE):
                    schema_index = beschikbare_indices[match.trainIdx]
                    bronpunten.append(symbool_keypoints[match.queryIdx].pt)
                    doelpunten.append(schema_keypoints[schema_index].pt)
                    schema_indices.append(schema_index)
        if len(bronpunten) < min_matches:
            break
        transformatie, inlier_masker = cv2.estimateAffinePartial2D(
            np.float32(bronpunten), np.float32(doelpunten),
            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_DREMPEL)
        if transformatie is None or int(inlier_masker.sum()) < min_matches:
            break
        beschikbaar[np.array(schema_indices)[inlier_masker.ravel() == 1]] = False   # niet opnieuw gebruiken
        hoogte, breedte = symbool_vorm
        omtrek = np.float32([[0, 0], [breedte, 0], [breedte, hoogte], [0, hoogte]]).reshape(-1, 1, 2)
        geprojecteerd = cv2.transform(omtrek, transformatie)
        box = cv2.boundingRect(geprojecteerd.astype(np.int32))
        exemplaren.append((box, min(1.0, float(inlier_masker.sum()) / len(symbool_keypoints))))
    return exemplaren


def zoek_symbolen(schema_grijs, symbolen, min_matches):
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    schema_keypoints, schema_descriptors = sift.detectAndCompute(schema_grijs, None)
    print(f"[info] schema: {len(schema_keypoints)} keypoints")
    if schema_descriptors is None:
        return []
    detecties = []
    for label, symbool_grijs in symbolen:
        symbool_keypoints, symbool_descriptors = sift.detectAndCompute(symbool_grijs, None)
        aantal = 0 if symbool_descriptors is None else len(symbool_keypoints)
        print(f"[info] {label}: {aantal} keypoints in het symbool")
        if aantal < max(2, min_matches):
            print(f"       te weinig keypoints (minimum {max(2, min_matches)}), overgeslagen")
            continue
        exemplaren = zoek_exemplaren(matcher, symbool_keypoints, symbool_descriptors, symbool_grijs.shape,
                                     schema_keypoints, schema_descriptors, min_matches)
        print(f"       {len(exemplaren)} exemplaar(en) gevonden")
        detecties += [{"label": label, "score": score, "box": box} for box, score in exemplaren]
    return detecties


def main():
    parser = argparse.ArgumentParser(description="Feature matching (SIFT, simpel)")
    parser.add_argument("schemas", nargs="+", help="een of meer afbeeldingen van schema's")
    parser.add_argument("--symbolen", default="symbolen_set1", help="map met de symbolen (standaard symbolen_set1)")
    parser.add_argument("--min-matches", type=int, default=10,
                        help="minimum aantal consistente matches (en keypoints) per exemplaar (standaard 10)")
    parser.add_argument("--uit", default="resultaten", help="map voor de uitvoer (standaard resultaten)")
    args = parser.parse_args()

    symbolen = laad_symbolen(args.symbolen)
    programma_naam = os.path.splitext(os.path.basename(__file__))[0]
    for schema_pad in args.schemas:
        schema = cv2.imread(schema_pad)
        if schema is None:
            print(f"Kan {schema_pad} niet lezen, overgeslagen")
            continue
        print(f"\n== {schema_pad} ==")
        start = time.perf_counter()
        detecties = zoek_symbolen(cv2.cvtColor(schema, cv2.COLOR_BGR2GRAY), symbolen, args.min_matches)
        sla_resultaat_op(programma_naam, schema_pad, schema, detecties, args.uit, time.perf_counter() - start)


if __name__ == "__main__":
    main()
