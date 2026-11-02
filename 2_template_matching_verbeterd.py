"""Programma 2 - Template matching, VERBETERDE versie.

Zelfde basis als programma 1 (grijswaarden, cv2.matchTemplate met TM_CCOEFF_NORMED, drempel), met precies
DRIE verbeteringen:

  1. Meerdere groottes   Elk symbool wordt op 37 groottes geprobeerd (0.15x tot 2.4x, stappen van 8%).
                         Reden: je uitgeknipte symbool is vaak groter/kleiner dan in het schema.
  2. NMS                 Non-Maximum Suppression: van een groep overlappende boxen blijft enkel de beste over
                         (IoU > --nms-iou, standaard 0.3). Reden: een symbool geeft veel naburige treffers.
  3. Omvattende boxen    Een box die voor >= 80% binnen een grotere box ligt, waarvan de score minstens 85% van
                         zijn eigen score is, wordt verwijderd. Reden: delen van een groot symbool lijken op een
                         klein symbool. (Dit gebeurt VOOR de NMS.)
  Elke verbetering kan uitgezet worden (--geen-schalen, --geen-nms, --geen-omvattend) voor je experimenten.

Gebruik
    python 2_template_matching_verbeterd.py schemas/schema1.png --symbolen symbolen_set1
    python 2_template_matching_verbeterd.py schemas/schema1.png --geen-omvattend      (zonder verbetering 3)

Uitvoer (in de map --uit, standaard "resultaten"): <programma>__<schema>.png en .txt
    txt: per regel label  x  y  breedte  hoogte  score   (pixels, x/y = linksboven)
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
SCHALEN = [0.15 * 1.08 ** stap for stap in range(37)]   # 0.15 ... ca. 2.4
MIN_ZIJDE = 10                                          # te kleine symbolen (pixels) worden overgeslagen
OMVATTEND_FRACTIE = 0.8                                 # verbetering 3: minimaal deel binnen de grotere box
OMVATTEND_SCOREVERHOUDING = 0.85                        # verbetering 3: score grotere box >= 85% van eigen score


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


def overlap_oppervlakte(box_a, box_b):
    """Oppervlakte (pixels) van de doorsnede van twee boxen (x, y, breedte, hoogte)."""
    x_a, y_a, breedte_a, hoogte_a = box_a
    x_b, y_b, breedte_b, hoogte_b = box_b
    overlap_breedte = max(0, min(x_a + breedte_a, x_b + breedte_b) - max(x_a, x_b))
    overlap_hoogte = max(0, min(y_a + hoogte_a, y_b + hoogte_b) - max(y_a, y_b))
    return overlap_breedte * overlap_hoogte


def intersection_over_union(box_a, box_b):
    doorsnede = overlap_oppervlakte(box_a, box_b)
    vereniging = box_a[2] * box_a[3] + box_b[2] * box_b[3] - doorsnede
    return doorsnede / vereniging if vereniging > 0 else 0.0


def verwijder_omvattende(detecties):
    """Verbetering 3: laat boxen weg die grotendeels in een grotere box met vergelijkbare score liggen."""
    behouden = []
    for kandidaat in detecties:
        _, _, breedte, hoogte = kandidaat["box"]
        is_fragment = False
        for andere in detecties:
            if andere is kandidaat or andere["box"][2] * andere["box"][3] <= breedte * hoogte:
                continue
            if andere["score"] < OMVATTEND_SCOREVERHOUDING * kandidaat["score"]:
                continue
            if overlap_oppervlakte(kandidaat["box"], andere["box"]) >= OMVATTEND_FRACTIE * breedte * hoogte:
                is_fragment = True
                break
        if not is_fragment:
            behouden.append(kandidaat)
    return behouden


def non_maximum_suppression(detecties, iou_drempel):
    """Verbetering 2: van overlappende boxen blijft de beste over (ook over verschillende labels heen)."""
    behouden = []
    for kandidaat in sorted(detecties, key=lambda detectie: detectie["score"], reverse=True):
        if all(intersection_over_union(kandidaat["box"], gekozen["box"]) <= iou_drempel for gekozen in behouden):
            behouden.append(kandidaat)
    return behouden


def zoek_symbolen(schema_grijs, symbolen, args):
    schalen = [1.0] if args.geen_schalen else SCHALEN
    detecties = []
    for label, symbool_grijs in symbolen:
        beste_score, beste_schaal = -1.0, None
        for schaal in schalen:
            geschaald = cv2.resize(symbool_grijs, None, fx=schaal, fy=schaal, interpolation=cv2.INTER_AREA)
            hoogte, breedte = geschaald.shape
            if (hoogte < MIN_ZIJDE or breedte < MIN_ZIJDE
                    or hoogte > schema_grijs.shape[0] or breedte > schema_grijs.shape[1]):
                continue
            if geschaald.std() < 1e-3:          # helemaal effen plaatje: correlatie is niet gedefinieerd
                continue
            scorekaart = cv2.matchTemplate(schema_grijs, geschaald, cv2.TM_CCOEFF_NORMED)
            if scorekaart.max() > beste_score:
                beste_score, beste_schaal = float(scorekaart.max()), schaal
            rijen, kolommen = np.where(scorekaart >= args.drempel)
            detecties += [{"label": label, "score": float(scorekaart[rij, kolom]),
                           "box": (int(kolom), int(rij), breedte, hoogte)}
                          for rij, kolom in zip(rijen, kolommen)]
        if beste_schaal is None:
            print(f"[info] {label}: geen bruikbare schaal")
        else:
            print(f"[info] {label}: hoogste score {beste_score:.2f} bij schaal {beste_schaal:.2f}")
    print(f"[info] ruwe treffers: {len(detecties)}")
    if not args.geen_omvattend:
        detecties = verwijder_omvattende(detecties)
        print(f"[info] na verwijderen omvattende boxen: {len(detecties)}")
    if not args.geen_nms:
        detecties = non_maximum_suppression(detecties, args.nms_iou)
        print(f"[info] na NMS: {len(detecties)}")
    return detecties


def main():
    parser = argparse.ArgumentParser(description="Template matching (verbeterd)")
    parser.add_argument("schemas", nargs="+", help="een of meer afbeeldingen van schema's")
    parser.add_argument("--symbolen", default="symbolen_set1", help="map met de symbolen (standaard symbolen_set1)")
    parser.add_argument("--drempel", type=float, default=0.6, help="minimale score 0-1 (standaard 0.6)")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="NMS: overlap vanaf deze IoU telt als dubbel")
    parser.add_argument("--geen-schalen", action="store_true", help="verbetering 1 uitzetten (enkel schaal 1.0)")
    parser.add_argument("--geen-nms", action="store_true", help="verbetering 2 uitzetten")
    parser.add_argument("--geen-omvattend", action="store_true", help="verbetering 3 uitzetten")
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
        detecties = zoek_symbolen(cv2.cvtColor(schema, cv2.COLOR_BGR2GRAY), symbolen, args)
        sla_resultaat_op(programma_naam, schema_pad, schema, detecties, args.uit, time.perf_counter() - start)


if __name__ == "__main__":
    main()
