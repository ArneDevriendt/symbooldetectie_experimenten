"""Programma 1 - Template matching, SIMPELE versie.

Wat dit programma doet
    Elk symbool (een uitgeknipt plaatje) wordt in zijn ORIGINELE grootte (schaal 1.0) over het schema
    geschoven. Op elke positie wordt de overeenkomst berekend (cv2.matchTemplate, TM_CCOEFF_NORMED,
    waarde tussen -1 en 1). Elke positie met een score >= drempel wordt gemeld.

Wat dit programma bewust NIET doet (dat zit in programma 2)
    - geen verschillende groottes - geen overlappende boxen weghalen - geen voorbewerking.
    Je krijgt dus ruwe resultaten: een symbool kan tientallen dicht bij elkaar liggende treffers geven.

Gebruik
    python 1_template_matching_simpel.py schemas/schema1.png --symbolen symbolen_set1
    python 1_template_matching_simpel.py schema_a.png schema_b.png --symbolen mijn_symbolen --drempel 0.7

Symbolen: elke afbeelding in de symbolenmap is een template; de bestandsnaam (zonder extensie) is het label.
    Meerdere voorbeelden van hetzelfde symbool: Klep_1.png, Klep_2.png -> label "Klep".

Uitvoer (in de map --uit, standaard "resultaten")
    <programma>__<schema>.png   het schema met gekleurde kaders (kleur per label)
    <programma>__<schema>.txt   per regel: label  x  y  breedte  hoogte  score   (pixels, x/y = linksboven)
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
    """Vaste kleur (BGR) per label, elke run dezelfde."""
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


def zoek_symbolen(schema_grijs, symbolen, drempel):
    """Template matching op schaal 1.0 voor elk symbool."""
    detecties = []
    for label, symbool_grijs in symbolen:
        symbool_hoogte, symbool_breedte = symbool_grijs.shape
        if symbool_hoogte > schema_grijs.shape[0] or symbool_breedte > schema_grijs.shape[1]:
            print(f"[info] {label}: symbool is groter dan het schema, overgeslagen")
            continue
        scorekaart = cv2.matchTemplate(schema_grijs, symbool_grijs, cv2.TM_CCOEFF_NORMED)
        print(f"[info] {label}: hoogste score {scorekaart.max():.2f}")
        rijen, kolommen = np.where(scorekaart >= drempel)
        for rij, kolom in zip(rijen, kolommen):
            detecties.append({"label": label, "score": float(scorekaart[rij, kolom]),
                              "box": (int(kolom), int(rij), symbool_breedte, symbool_hoogte)})
    return detecties


def main():
    parser = argparse.ArgumentParser(description="Template matching (simpel)")
    parser.add_argument("schemas", nargs="+", help="een of meer afbeeldingen van schema's")
    parser.add_argument("--symbolen", default="symbolen_set1", help="map met de symbolen (standaard symbolen_set1)")
    parser.add_argument("--drempel", type=float, default=0.6, help="minimale score 0-1 (standaard 0.6)")
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
        detecties = zoek_symbolen(cv2.cvtColor(schema, cv2.COLOR_BGR2GRAY), symbolen, args.drempel)
        sla_resultaat_op(programma_naam, schema_pad, schema, detecties, args.uit, time.perf_counter() - start)


if __name__ == "__main__":
    main()
