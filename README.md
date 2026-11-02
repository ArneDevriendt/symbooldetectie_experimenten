# Symbooldetectie – experimenten

Vijf losstaande programma's om symbolen te lokaliseren op een industrieel schema.
Elk programma is één .py-bestand en deelt niets met de andere.

| # | Bestand | Wat |
|---|---------|-----|
| 1 | `1_template_matching_simpel.py` | Template matching, één schaal, geen nabewerking |
| 2 | `2_template_matching_verbeterd.py` | Zelfde + multi-schaal, verwijderen van omvattende boxen, NMS |
| 3 | `3_feature_matching_simpel.py` | SIFT + afstandstolerantie + RANSAC, vindt alle exemplaren |
| 4 | `4_yolo_trainen.py` | YOLOv8 trainen (CPU of GPU), toont mAP op het einde |
| 5 | `5_yolo_testen.py` | Getraind YOLOv8-model toepassen op schema's |

## Installatie (Windows PowerShell)

```powershell
python -m venv venv          # Python 3.12 aanbevolen
.\venv\Scripts\activate
pip install -r requirements.txt
```

Voor training op de GPU: installeer eerst PyTorch met CUDA (zie pytorch.org), daarna de rest.

## Mappen

- `schemas/` – schema's om te testen (`schema1.png` als voorbeeld)
- `symbolen_set1/` – uitgeknipte symbolen; de bestandsnaam (zonder extensie) is het label
- Zet een tweede set bv. in `symbolen_set2/` en geef die mee met `--symbolen`

## Programma 1, 2 en 3

```powershell
python 1_template_matching_simpel.py schemas\schema1.png --symbolen symbolen_set1
python 2_template_matching_verbeterd.py schemas\schema1.png --symbolen symbolen_set1
python 3_feature_matching_simpel.py schemas\schema1.png --symbolen symbolen_set1
```

Opties: `--drempel` (1 en 2, standaard 0.6), `--min-matches` (3, standaard 10), `--uit` (uitvoermap).
Programma 2 heeft `--geen-schalen`, `--geen-omvattend`, `--geen-nms` en `--nms-iou`.

## Programma 4 – trainen

Je hebt een YOLO-dataset nodig (`images/train`, `images/val`, `labels/train`, `labels/val`) en een `data.yaml`:

```yaml
path: C:/pad/naar/dataset
train: images/train
val: images/val
names:
  0: Lift
  1: Silo
```

```powershell
python 4_yolo_trainen.py --data data.yaml --naam yolo_cpu --device cpu
python 4_yolo_trainen.py --data data.yaml --naam yolo_gpu --device 0
```

Opties: `--model`, `--epochs` (100), `--imgsz` (1280), `--batch` (4), `--seed`.
Uitvoer: trainingstijd, mAP50, mAP50-95, precision en recall (totaal en per klasse) in `runs/<naam>/metrics.txt`; het model in `weights/<naam>.pt`.

## Programma 5 – testen

```powershell
python 5_yolo_testen.py schemas\schema1.png --gewichten weights\yolo_gpu.pt
```

Opties: `--conf` (0.25), `--iou` (0.3), `--nested` (0.6; 0 = uit), `--imgsz` (zelfde als bij training), `--device`.

## Uitvoer

Alle programma's schrijven naar `resultaten/`: een afbeelding met gekleurde boxes per label (`<programma>__<schema>.png`) en een tekstbestand met per lijn, tab-gescheiden:
`label  x  y  breedte  hoogte  score`
