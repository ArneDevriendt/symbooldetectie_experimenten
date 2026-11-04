"""Programma 3b - Feature matching (SIFT), VERBETERDE versie.

Zelfde basis als programma 3 (SIFT, kandidaat-matches met afstandstolerantie, RANSAC, gebruikte keypoints
uitsluiten), plus precies 7 eenvoudige verbeteringen die elk apart uitgezet kunnen worden:

    1. OPSCHALEN        Schema en symbool worden vergroot (schema x2, symbool minstens 200 px). Dunne lijnen
                        geven zo meer keypoints.                                        (uitzetten: --geen-opschalen)
    2. GEVOELIGERE SIFT contrastThreshold 0.005 (standaard 0.04) en edgeThreshold 30 (standaard 10): meer
                        keypoints op dunne omtrekken zonder veel textuur.                (uitzetten: --geen-gevoelig)
    3. SCHAALFILTER     Een plaatsing waarbij het symbool kleiner dan 0.12x of groter dan 3x zijn originele grootte
                        zou zijn, wordt verworpen (RANSAC vindt anders "boxen" van 1 pixel: alle matches
                        samengeklapt op één punt).                                   (uitzetten: --geen-schaalfilter)
    4. EEN-OP-EEN       Elk keypoint van het schema mag per ronde maar één keer meetellen (de beste match).
                        Anders matchen veel keypoints van het symbool op dezelfde paar schemapunten en klapt
                        RANSAC alles samen tot grootte 0.                           (uitzetten: --geen-een-op-een)
    5. MEERDERE         Het symbool wordt op verschillende groottes opnieuw geprobeerd. SIFT is in theorie
       GROOTTES         schaalonafhankelijk, maar in de praktijk matcht een symbool dat 5x groter is dan in het
                        schema slecht (bv. een uitgeknipte Klepkast).               (uitzetten: --geen-groottes)
    6. MINIMUMSCORE     Detecties met score < --min-score (standaard 0.10) worden weggelaten; toevalstreffers
                        hebben een lage score.                                      (uitzetten: --min-score 0)
    7. NMS              Overlappende boxen (IoU > --nms-iou) van het einde worden samengevoegd tot de beste.
                                                                                        (uitzetten: --geen-nms)

Gebruik
    python 3b_feature_matching_verbeterd.py schemas/schema1.png --symbolen symbolen_set1
    python 3b_feature_matching_verbeterd.py schema.png --symbolen symbolen_set2 --geen-nms

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

SCHEMA_SCHAAL = 3.0          # verbetering 1: vergroting van het schema (en minimum voor het symbool)
MIN_SYMBOOL_ZIJDE = 200      # verbetering 1: korte zijde van het symbool wordt minstens zo groot
VERVAGING = 3                # verbetering 1: lichte blur na het vergroten (oneven getal)
CONTRAST_DREMPEL = 0.005     # verbetering 2: SIFT-standaard is 0.04
RAND_DREMPEL = 30            # verbetering 2: SIFT-standaard is 10
SYMBOOL_GROOTTES = (0.1, 0.15, 0.22, 0.33, 0.5, 0.75, 1.0)   # verbetering 5: factor t.o.v. de schemaschaal
SCHAALBEREIK = (0.12, 3.0)   # verbetering 3: toegelaten grootte t.o.v. het originele symbool


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


def vergroot(grijs, schaal):
    """Vergroot en vervaag licht (verbetering 1). Bij schaal 1.0 blijft de afbeelding ongewijzigd."""
    if schaal == 1.0:
        return grijs
    vergroot_beeld = cv2.resize(grijs, None, fx=schaal, fy=schaal, interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(vergroot_beeld, (VERVAGING, VERVAGING), 0)


def intersection_over_union(box_a, box_b):
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    snijding = max(0, min(xa + wa, xb + wb) - max(xa, xb)) * max(0, min(ya + ha, yb + hb) - max(ya, yb))
    unie = wa * ha + wb * hb - snijding
    return snijding / unie if unie > 0 else 0.0


def non_maximum_suppression(detecties, iou_drempel):
    """Verbetering 7: behoud per groep overlappende boxen enkel de beste score."""
    behouden = []
    for kandidaat in sorted(detecties, key=lambda d: d["score"], reverse=True):
        if all(intersection_over_union(kandidaat["box"], gekozen["box"]) <= iou_drempel for gekozen in behouden):
            behouden.append(kandidaat)
    return behouden


def zoek_exemplaren(matcher, symbool_keypoints, symbool_descriptors, symbool_vorm, symbool_schaal, schema_schaal,
                    schema_keypoints, schema_descriptors, min_matches, gebruik_schaalfilter, een_op_een):
    """Alle exemplaren van een symbool: lijst van ((x, y, breedte, hoogte), score).
    symbool_vorm = (hoogte, breedte) van het (eventueel vergrote) symbool."""
    beschikbaar = np.ones(len(schema_keypoints), dtype=bool)       # nog niet gebruikte keypoints van het schema
    exemplaren = []
    for _ in range(MAX_EXEMPLAREN):
        beschikbare_indices = np.flatnonzero(beschikbaar)
        if len(beschikbare_indices) < min_matches:
            break
        kandidaat_lijsten = matcher.knnMatch(symbool_descriptors, schema_descriptors[beschikbare_indices],
                                             k=min(CANDIDATEN, len(beschikbare_indices)))
        goede_matches = [match for kandidaten in kandidaat_lijsten for match in kandidaten
                         if match.distance <= kandidaten[0].distance * (1 + AFSTANDSTOLERANTIE)]
        if een_op_een:   # verbetering 4: per schemapunt enkel de beste match
            beste_per_schemapunt = {}
            for match in goede_matches:
                huidige = beste_per_schemapunt.get(match.trainIdx)
                if huidige is None or match.distance < huidige.distance:
                    beste_per_schemapunt[match.trainIdx] = match
            goede_matches = list(beste_per_schemapunt.values())
        bronpunten = [symbool_keypoints[match.queryIdx].pt for match in goede_matches]
        schema_indices = [beschikbare_indices[match.trainIdx] for match in goede_matches]
        doelpunten = [schema_keypoints[schema_index].pt for schema_index in schema_indices]
        if len(bronpunten) < min_matches:
            break
        transformatie, inlier_masker = cv2.estimateAffinePartial2D(
            np.float32(bronpunten), np.float32(doelpunten),
            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_DREMPEL)
        if transformatie is None or int(inlier_masker.sum()) < min_matches:
            break
        beschikbaar[np.array(schema_indices)[inlier_masker.ravel() == 1]] = False   # niet opnieuw gebruiken
        if gebruik_schaalfilter:
            relatieve_grootte = np.hypot(transformatie[0, 0], transformatie[1, 0]) * symbool_schaal / schema_schaal
            if not SCHAALBEREIK[0] <= relatieve_grootte <= SCHAALBEREIK[1]:
                continue   # onmogelijke grootte: keypoints blijven uitgesloten, zoek verder
        hoogte, breedte = symbool_vorm
        omtrek = np.float32([[0, 0], [breedte, 0], [breedte, hoogte], [0, hoogte]]).reshape(-1, 1, 2)
        geprojecteerd = cv2.transform(omtrek, transformatie) / schema_schaal   # terug naar originele pixels
        box = cv2.boundingRect(geprojecteerd.astype(np.int32))
        exemplaren.append((box, min(1.0, float(inlier_masker.sum()) / len(symbool_keypoints))))
    return exemplaren


def zoek_symbolen(schema_grijs, symbolen, args):
    schema_schaal = 1.0 if args.geen_opschalen else SCHEMA_SCHAAL
    if args.geen_gevoelig:
        sift = cv2.SIFT_create()
    else:
        sift = cv2.SIFT_create(contrastThreshold=CONTRAST_DREMPEL, edgeThreshold=RAND_DREMPEL)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    schema_keypoints, schema_descriptors = sift.detectAndCompute(vergroot(schema_grijs, schema_schaal), None)
    print(f"[info] schema: {len(schema_keypoints)} keypoints")
    if schema_descriptors is None:
        return []
    detecties = []
    for label, symbool_grijs in symbolen:
        if args.geen_opschalen:
            symbool_schalen = [1.0]
        elif args.geen_groottes:
            symbool_schalen = [max(schema_schaal, MIN_SYMBOOL_ZIJDE / min(symbool_grijs.shape))]
        else:
            symbool_schalen = [schema_schaal * factor for factor in SYMBOOL_GROOTTES]
        aantal_exemplaren = 0
        for symbool_schaal in symbool_schalen:
            symbool = vergroot(symbool_grijs, symbool_schaal)
            symbool_keypoints, symbool_descriptors = sift.detectAndCompute(symbool, None)
            aantal = 0 if symbool_descriptors is None else len(symbool_keypoints)
            if aantal < max(2, args.min_matches):
                continue
            # Het symbool is x symbool_schaal vergroot en het schema x schema_schaal; de transformatie bevat dus
            # beide factoren, en de box wordt in zoek_exemplaren door schema_schaal gedeeld.
            exemplaren = zoek_exemplaren(matcher, symbool_keypoints, symbool_descriptors, symbool.shape,
                                         symbool_schaal, schema_schaal, schema_keypoints, schema_descriptors,
                                         args.min_matches, not args.geen_schaalfilter, not args.geen_een_op_een)
            print(f"[info] {label} (symbool x{symbool_schaal:.2f}, {aantal} keypoints): {len(exemplaren)} exemplaar(en)")
            aantal_exemplaren += len(exemplaren)
            detecties += [{"label": label, "score": score, "box": box} for box, score in exemplaren]
        if aantal_exemplaren == 0:
            print(f"[info] {label}: niets gevonden")
    print(f"[info] ruwe detecties: {len(detecties)}")
    detecties = [detectie for detectie in detecties if detectie["score"] >= args.min_score]
    print(f"[info] na minimumscore: {len(detecties)}")
    if not args.geen_nms:
        detecties = non_maximum_suppression(detecties, args.nms_iou)
        print(f"[info] na NMS: {len(detecties)}")
    return detecties


def main():
    parser = argparse.ArgumentParser(description="Feature matching (SIFT, verbeterd)")
    parser.add_argument("schemas", nargs="+", help="een of meer afbeeldingen van schema's")
    parser.add_argument("--symbolen", default="symbolen_set1", help="map met de symbolen (standaard symbolen_set1)")
    parser.add_argument("--min-matches", type=int, default=10,
                        help="minimum aantal consistente matches (en keypoints) per exemplaar (standaard 10)")
    parser.add_argument("--min-score", type=float, default=0.10, help="verbetering 6: minimumscore (standaard 0.10; 0 = uit)")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="IoU-drempel van de NMS (standaard 0.3)")
    parser.add_argument("--geen-opschalen", action="store_true", help="zet verbetering 1 (opschalen) uit")
    parser.add_argument("--geen-gevoelig", action="store_true", help="zet verbetering 2 (gevoeligere SIFT) uit")
    parser.add_argument("--geen-schaalfilter", action="store_true", help="zet verbetering 3 (schaalfilter) uit")
    parser.add_argument("--geen-groottes", action="store_true", help="zet verbetering 5 (meerdere groottes) uit")
    parser.add_argument("--geen-een-op-een", action="store_true", help="zet verbetering 4 (een-op-een matching) uit")
    parser.add_argument("--geen-nms", action="store_true", help="zet verbetering 7 (NMS) uit")
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
