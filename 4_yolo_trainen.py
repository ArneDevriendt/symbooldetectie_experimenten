"""Programma 4 - YOLOv8 TRAINEN (en de mAP van het getrainde model tonen).

Wat het doet
    1. Traint een YOLOv8-model (ultralytics) op jouw gelabelde schema's, beschreven in een data.yaml.
    2. Meet de tijd van de training (handig om CPU en GPU te vergelijken).
    3. Valideert het beste model op de validatieset en print mAP50, mAP50-95, precision en recall
       (algemeen en per klasse). De cijfers worden ook bewaard in runs/<naam>/metrics.txt.
    4. Kopieert het beste model naar weights/<naam>.pt, zodat programma 5 het kan gebruiken.

Verwachte datasetstructuur (zoals je al had)
    dataset/
        images/train/*.png   images/val/*.png
        labels/train/*.txt   labels/val/*.txt      (YOLO-formaat: klasse x_midden y_midden breedte hoogte, 0-1)
    data.yaml:
        path: C:/pad/naar/dataset        (absoluut pad is het veiligst)
        train: images/train
        val: images/val
        names:
          0: Silo
          1: Klep
          ...

Gebruik (voorbeelden)
    python 4_yolo_trainen.py --data data.yaml --device cpu --naam yolo_cpu
    python 4_yolo_trainen.py --data data.yaml --device 0   --naam yolo_gpu --epochs 200 --model yolov8s.pt

    --device: "cpu" of het nummer van de GPU (0). Standaard "auto" (ultralytics kiest zelf).
    Voor een eerlijke CPU/GPU-vergelijking: alles gelijk houden behalve --device (ook --seed).
    --imgsz bij het trainen moet je ook bij het testen (programma 5) gebruiken.
"""
import argparse
import os
import shutil
import time

# Op Windows starten de dataloader-workers dit bestand opnieuw op: alles staat daarom in main()
# en main() wordt enkel uitgevoerd onder "if __name__ == '__main__'" (onderaan).


def schrijf_metrics(pad, naam, args, trainingsseconden, metrics, klassennamen):
    regels = [
        f"naam: {naam}",
        f"model: {args.model}   epochs: {args.epochs}   imgsz: {args.imgsz}   batch: {args.batch}   "
        f"device: {args.device}   seed: {args.seed}",
        f"trainingstijd: {trainingsseconden / 60:.1f} min ({trainingsseconden:.0f} s)",
        f"mAP50:    {metrics.box.map50:.4f}",
        f"mAP50-95: {metrics.box.map:.4f}",
        f"precision (gemiddeld): {metrics.box.mp:.4f}",
        f"recall (gemiddeld):    {metrics.box.mr:.4f}",
        "",
        f"{'klasse':<20}{'mAP50':>8}{'mAP50-95':>10}",
    ]
    for klasse_index, mAP50, mAP50_95 in zip(metrics.box.ap_class_index, metrics.box.ap50, metrics.box.ap):
        regels.append(f"{klassennamen[int(klasse_index)]:<20}{mAP50:>8.4f}{mAP50_95:>10.4f}")
    tekst = "\n".join(regels)
    print("\n" + tekst)
    with open(pad, "w", encoding="utf-8") as bestand:
        bestand.write(tekst + "\n")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 trainen en mAP tonen")
    parser.add_argument("--data", default="data.yaml", help="pad naar data.yaml (standaard data.yaml)")
    parser.add_argument("--naam", default="yolo", help="naam van de run en van het gewichtenbestand")
    parser.add_argument("--model", default="yolov8n.pt", help="startmodel: yolov8n/s/m/l/x.pt (standaard n)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280, help="beeldgrootte; schema's zijn groot met kleine symbolen")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto", help='"cpu", "0" (eerste GPU) of "auto"')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise SystemExit(f"data.yaml niet gevonden: {args.data}")

    from ultralytics import YOLO           # hier geimporteerd zodat --help snel werkt

    device = args.device
    if device not in ("auto", "cpu"):
        device = int(device)
    if device == "auto":
        device = None                       # ultralytics kiest zelf (GPU indien beschikbaar)

    runs_map = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    model = YOLO(args.model)
    start = time.perf_counter()
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
                seed=args.seed, project=runs_map, name=args.naam, exist_ok=True)
    trainingsseconden = time.perf_counter() - start

    beste_gewichten = os.path.join(runs_map, args.naam, "weights", "best.pt")
    beste_model = YOLO(beste_gewichten)
    metrics = beste_model.val(data=args.data, imgsz=args.imgsz, device=device, verbose=False)
    schrijf_metrics(os.path.join(runs_map, args.naam, "metrics.txt"), args.naam, args, trainingsseconden,
                    metrics, beste_model.names)

    gewichten_map = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(gewichten_map, exist_ok=True)
    doel = os.path.join(gewichten_map, f"{args.naam}.pt")
    shutil.copy(beste_gewichten, doel)
    print(f"\nGewichten bewaard in {doel}")
    print(f"Testen: python 5_yolo_testen.py schemas/schema1.png --gewichten {doel} --imgsz {args.imgsz}")


if __name__ == "__main__":
    main()
