"""Programma 5 - YOLOv8 TESTEN: symbolen vinden in een schema met een getraind model.

Gebruik
    python 5_yolo_testen.py schemas/schema1.png --gewichten weights/yolo_gpu.pt
    python 5_yolo_testen.py schema_a.png schema_b.png --gewichten weights/yolo_cpu.pt --imgsz 1280 --device cpu

Belangrijk: --imgsz moet gelijk zijn aan de waarde waarmee je getraind hebt (programma 4, standaard 1280).

Instellingen (zoals in je oude app.py)
    --conf    minimale zekerheid van het model (standaard 0.25)
    --iou     YOLO's eigen overlapfilter (NMS). Standaard 0.3 (de ultralytics-standaard 0.7 laat te veel dubbele
              kaders door).
    --nested  extra filter: een kader van dezelfde klasse waarvan meer dan deze fractie (standaard 0.6) in een
              beter kader ligt, wordt verwijderd. 0 = dit filter uitzetten.

Uitvoer (in de map --uit, standaard "resultaten"): <programma>__<schema>.png en .txt
    txt: per regel label  x  y  breedte  hoogte  score   (pixels, x/y = linksboven)
"""
import argparse
import colorsys
import os
import time
import zlib

import cv2


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


def verwijder_geneste(detecties, nested_fractie):
    """Per klasse: behoud de beste box, laat boxen weg die grotendeels in (of over) een betere box liggen."""
    behouden = []
    for kandidaat in sorted(detecties, key=lambda detectie: detectie["score"], reverse=True):
        x, y, breedte, hoogte = kandidaat["box"]
        is_dubbel = False
        for gekozen in behouden:
            if gekozen["label"] != kandidaat["label"]:
                continue
            gekozen_x, gekozen_y, gekozen_breedte, gekozen_hoogte = gekozen["box"]
            doorsnede = (max(0, min(x + breedte, gekozen_x + gekozen_breedte) - max(x, gekozen_x))
                         * max(0, min(y + hoogte, gekozen_y + gekozen_hoogte) - max(y, gekozen_y)))
            kleinste_oppervlakte = max(1, min(breedte * hoogte, gekozen_breedte * gekozen_hoogte))
            if doorsnede / kleinste_oppervlakte > nested_fractie:
                is_dubbel = True
                break
        if not is_dubbel:
            behouden.append(kandidaat)
    return behouden


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 testen op schema's")
    parser.add_argument("schemas", nargs="+", help="een of meer afbeeldingen van schema's")
    parser.add_argument("--gewichten", default="weights/yolo.pt", help="getraind model (.pt) uit programma 4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--nested", type=float, default=0.6, help="0 = extra filter uitzetten")
    parser.add_argument("--imgsz", type=int, default=1280, help="gelijk aan de waarde bij het trainen")
    parser.add_argument("--device", default="auto", help='"cpu", "0" of "auto"')
    parser.add_argument("--uit", default="resultaten", help="map voor de uitvoer (standaard resultaten)")
    args = parser.parse_args()

    if not os.path.isfile(args.gewichten):
        raise SystemExit(f"Gewichten niet gevonden: {args.gewichten} (eerst trainen met 4_yolo_trainen.py)")
    from ultralytics import YOLO

    device = None if args.device == "auto" else (args.device if args.device == "cpu" else int(args.device))
    model = YOLO(args.gewichten)
    programma_naam = os.path.splitext(os.path.basename(__file__))[0] + "_" + \
        os.path.splitext(os.path.basename(args.gewichten))[0]

    for schema_pad in args.schemas:
        schema = cv2.imread(schema_pad)
        if schema is None:
            print(f"Kan {schema_pad} niet lezen, overgeslagen")
            continue
        print(f"\n== {schema_pad} ==")
        start = time.perf_counter()
        resultaat = model.predict(schema, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                  device=device, verbose=False)[0]
        detecties = []
        for kader in resultaat.boxes:
            x1, y1, x2, y2 = kader.xyxy[0].tolist()
            detecties.append({"label": resultaat.names[int(kader.cls)], "score": float(kader.conf),
                              "box": (x1, y1, x2 - x1, y2 - y1)})
        if args.nested > 0:
            detecties = verwijder_geneste(detecties, args.nested)
        sla_resultaat_op(programma_naam, schema_pad, schema, detecties, args.uit, time.perf_counter() - start)


if __name__ == "__main__":
    main()
