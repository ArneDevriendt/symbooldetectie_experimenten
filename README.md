# Symbooldetectie op industriële schema's – experimenten

Dit project bevat zes losstaande programma's die symbolen (bv. Silo, Klep, Lift) lokaliseren op een afbeelding van een
industrieel schema. Er zijn drie technieken:

| Techniek | Principe |
|---|---|
| **Template matching** | Een uitgeknipt symbool wordt pixel per pixel over het schema geschoven; waar de overeenkomst (score) hoog genoeg is, ligt het symbool. |
| **Feature matching (SIFT)** | Van symbool en schema worden herkenbare punten (keypoints) met een beschrijving (descriptor) berekend. Komen genoeg punten op een consistente manier overeen, dan is het symbool gevonden. |
| **YOLOv8 (deep learning)** | Een neuraal netwerk wordt getraind op gelabelde schema's en voorspelt daarna zelf de boxen en klassen. |

Dit document beschrijft enkel **wat** de programma's doen en **hoe** je ze gebruikt. Resultaten en conclusies staan
bewust niet in dit bestand.

---

## 1. Overzicht van de programma's

Elk programma is één zelfstandig `.py`-bestand. Ze delen geen code met elkaar (sommige hulpfuncties staan daarom
meermaals in de bestanden). Je kan dus elk programma apart gebruiken of verwijderen.

| Bestand | Techniek | Wat |
|---|---|---|
| `1_template_matching_simpel.py` | Template matching | Eén schaal, één drempel, geen nabewerking. |
| `2_template_matching_verbeterd.py` | Template matching | Zelfde basis + 3 verbeteringen (zie 4.2). |
| `3_feature_matching_simpel.py` | Feature matching | SIFT met standaardinstellingen, RANSAC, zoekt alle exemplaren. |
| `3b_feature_matching_verbeterd.py` | Feature matching | Zelfde basis + 7 verbeteringen (zie 4.4). |
| `4_yolo_trainen.py` | YOLOv8 | Trainen op CPU of GPU, toont de mAP op het einde. |
| `5_yolo_testen.py` | YOLOv8 | Een getraind model toepassen op schema's. |

---

## 2. Mappenstructuur

```
symbooldetectie_experimenten/
├── 1_template_matching_simpel.py
├── 2_template_matching_verbeterd.py
├── 3_feature_matching_simpel.py
├── 3b_feature_matching_verbeterd.py
├── 4_yolo_trainen.py
├── 5_yolo_testen.py
├── requirements.txt
├── .gitignore
├── schemas_set1/        schema's om te testen (set 1)
├── schemas_set2/        schema's om te testen (set 2)
├── symbolen_set1/       uitgeknipte referentiesymbolen, set 1 (Lift, Silo, Ketting, Klep, Klepkast)
├── symbolen_set2/       uitgeknipte referentiesymbolen, set 2
├── dataset/             YOLO-trainingsdata
│   ├── data.yaml
│   ├── images/train/    trainingsafbeeldingen
│   ├── images/val/      validatieafbeeldingen
│   ├── labels/train/    YOLO-labels bij de trainingsafbeeldingen (.txt)
│   └── labels/val/      YOLO-labels bij de validatieafbeeldingen (.txt)
├── resultaten/          uitvoer van programma 1, 2, 3, 3b en 5 (aangemaakt bij het draaien)
├── runs/                trainingsuitvoer van programma 4 (grafieken, metrics.txt, last.pt/best.pt)
└── weights/             getrainde modellen: <naam>.pt (kopie van best.pt)
```

**Hoe de mappen samenhangen**

- De **bestandsnaam** van een symbool is het **label**: `symbolen_set1/Silo.png` geeft het label `Silo`.
  Meerdere voorbeelden van hetzelfde symbool mogen: `Klep_1.png` en `Klep_2.png` tellen allebei als `Klep`.
- Programma 1, 2, 3 en 3b lezen een symbolenmap (`--symbolen`) en één of meer schema's.
- Programma 4 leest `dataset/data.yaml` en schrijft naar `runs/<naam>/` en `weights/<naam>.pt`.
- Programma 5 leest `weights/<naam>.pt` en schrijft naar `resultaten/`.

---

## 3. Installatie (Windows PowerShell)

1. **Python 3.12** installeren (nieuwere versies kunnen problemen geven met sommige pakketten). Controleer met
   `python --version`.
2. Virtuele omgeving maken en activeren:
   ```powershell
   cd C:\pad\naar\symbooldetectie_experimenten
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Enkel voor training op de GPU:** installeer eerst PyTorch met CUDA (kies op pytorch.org de regel die bij je
   CUDA-versie past; `nvidia-smi` toont je driver). Bijvoorbeeld:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
4. De rest installeren:
   ```powershell
   pip install -r requirements.txt
   ```
5. GPU controleren (optioneel):
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   `True` betekent dat PyTorch de GPU kan gebruiken. In de versienaam staat niet altijd `+cu...`.

`requirements.txt` bevat `numpy`, `opencv-python` en `ultralytics`. Programma 1, 2, 3 en 3b hebben enkel `numpy` en
`opencv-python` nodig. De versie van `ultralytics` (en dus van `torch`) kan de uitvoer van YOLO beïnvloeden; noteer die
bij je experimenten (`pip list`).

---

## 4. Gebruik per programma

Alle programma's schrijven hun uitvoer naar `resultaten/` (zie hoofdstuk 5) en tonen op de terminal wat ze gevonden
hebben. Je kan meerdere schema's tegelijk meegeven.

### 4.1 Programma 1 – Template matching, simpel

```powershell
python 1_template_matching_simpel.py schemas_set1\Schema1.png --symbolen symbolen_set1
```

**Werking:** schema en symbool worden grijs gemaakt. `cv2.matchTemplate` (methode `TM_CCOEFF_NORMED`) berekent voor elke
positie een score tussen -1 en 1. Posities boven de drempel zijn detecties. Het symbool wordt **op zijn originele
grootte** gezocht, er is geen nabewerking.

| Optie | Standaard | Betekenis |
|---|---|---|
| `--symbolen` | `symbolen_set1` | Map met de referentiesymbolen. |
| `--drempel` | `0.6` | Minimale score (0–1). Lager = meer treffers, meer fouten. |
| `--uit` | `resultaten` | Uitvoermap. |

Een symbool dat groter is dan het schema wordt overgeslagen (met een melding).

### 4.2 Programma 2 – Template matching, verbeterd

```powershell
python 2_template_matching_verbeterd.py schemas_set1\Schema1.png --symbolen symbolen_set1
```

Zelfde basis als programma 1, met precies drie verbeteringen. Elke verbetering kan apart uitgezet worden, zodat je het
effect ervan kan afzonderen:

| # | Verbetering | Wat het doet | Uitzetten |
|---|---|---|---|
| 1 | Meerdere groottes | Elk symbool wordt op 37 groottes geprobeerd (0.15× tot 2.4×, stappen van 8 %). | `--geen-schalen` |
| 2 | Omvattende boxen verwijderen | Een box die voor ≥ 80 % binnen een grotere box ligt met minstens 85 % van zijn score, wordt verwijderd (fragmenten van een groot symbool). Gebeurt vóór NMS. | `--geen-omvattend` |
| 3 | NMS (non-maximum suppression) | Van boxen die elkaar overlappen (IoU > `--nms-iou`) blijft enkel de beste over. | `--geen-nms` |

Extra opties: `--drempel` (0.6), `--nms-iou` (0.3), `--symbolen`, `--uit`. Het programma toont per label de hoogste
score en de bijhorende schaal, en hoeveel treffers er na elke stap overblijven.

### 4.3 Programma 3 – Feature matching, simpel

```powershell
python 3_feature_matching_simpel.py schemas_set1\Schema1.png --symbolen symbolen_set1
```

**Werking per symbool:**

1. SIFT (standaardinstellingen) berekent keypoints en descriptors voor symbool en schema.
2. Voor elk keypoint van het symbool worden de 4 beste kandidaat-matches in het schema bewaard, met alle kandidaten die
   hoogstens 30 % slechter zijn dan de beste. (Lowe's ratio-test werkt hier niet: als een symbool meerdere keren
   voorkomt, zijn de beste en de tweede beste match even goed.)
3. RANSAC (`cv2.estimateAffinePartial2D`) zoekt de grootste groep matches die bij één plaatsing past (verschuiving,
   schaal, rotatie). Dat is één exemplaar.
4. De omtrek van het symbool wordt met die transformatie in het schema geprojecteerd; dat geeft de box.
5. De gebruikte schema-keypoints worden uitgesloten en de zoektocht herhaalt tot er te weinig matches zijn
   (maximaal 50 exemplaren per symbool).

| Optie | Standaard | Betekenis |
|---|---|---|
| `--min-matches` | `10` | Minimum aantal consistente matches (en keypoints in het symbool) per exemplaar. |
| `--symbolen`, `--uit` | | Zoals hierboven. |

Het programma toont het aantal keypoints per symbool. Symbolen met te weinig keypoints worden overgeslagen. De score in
het txt-bestand is het aantal inliers gedeeld door het aantal keypoints van het symbool (maximaal 1).

### 4.4 Programma 3b – Feature matching, verbeterd

```powershell
python 3b_feature_matching_verbeterd.py schemas_set1\Schema1.png --symbolen symbolen_set1
```

Zelfde basis als programma 3, met extra verbeteringen die elk apart uitgezet kunnen worden:

| # | Verbetering | Wat het doet | Uitzetten |
|---|---|---|---|
| 1 | Opschalen | Schema (×3) en symbool worden vergroot en licht vervaagd, zodat dunne lijnen meer keypoints geven. | `--geen-opschalen` |
| 2 | Gevoeligere SIFT | `contrastThreshold` 0.005 (standaard 0.04) en `edgeThreshold` 30 (standaard 10). | `--geen-gevoelig` |
| 3 | Schaalfilter | Plaatsingen waarbij het symbool kleiner dan 0.12× of groter dan 3× zijn originele grootte zou zijn, worden verworpen. | `--geen-schaalfilter` |
| 4 | Eén-op-één matching | Elk schema-keypoint telt per ronde maar één keer mee (de beste match). Zonder dit klapt RANSAC samen op enkele punten. | `--geen-een-op-een` |
| 5 | Meerdere symboolgroottes | Het symbool wordt op 7 groottes geprobeerd. | `--geen-groottes` |
| 6 | Minimumscore | Detecties met score < `--min-score` (standaard 0.10) vallen weg. | `--min-score 0` |
| 7 | NMS | Overlappende boxen (IoU > `--nms-iou`, standaard 0.3) worden samengevoegd tot de beste. | `--geen-nms` |

Overige opties: `--min-matches` (10), `--symbolen`, `--uit`. Het programma toont per label en per grootte hoeveel
exemplaren gevonden zijn, en hoeveel detecties na de minimumscore en na NMS overblijven.

### 4.5 Programma 4 – YOLOv8 trainen

**Dataset (YOLO-formaat)**

- Per afbeelding in `dataset/images/<train|val>/` hoort een `.txt` met dezelfde naam in `dataset/labels/<train|val>/`.
- Elke regel in een label-bestand is één symbool: `klasse x_midden y_midden breedte hoogte`, alle waarden tussen 0 en 1
  (relatief aan de afbeelding).
- `dataset/data.yaml` koppelt alles:
  ```yaml
  path: C:/pad/naar/symbooldetectie_experimenten/dataset
  train: images/train
  val: images/val
  names:
    0: Lift
    1: Silo
    2: Ketting
    3: Klep
    4: Klepkast
  ```
  Gebruik een **absoluut pad** bij `path`. Ultralytics plakt `path` en `train`/`val` aan elkaar; een fout `path` geeft
  "images not found".
- De testschema's mogen **niet** in `dataset/` zitten.

**Starten**

```powershell
python 4_yolo_trainen.py --data dataset\data.yaml --naam yolo_cpu --device cpu --epochs 10 --imgsz 640 --batch 4 --seed 0
python 4_yolo_trainen.py --data dataset\data.yaml --naam yolo_gpu --device 0   --epochs 10 --imgsz 640 --batch 4 --seed 0
```

| Optie | Standaard | Betekenis |
|---|---|---|
| `--data` | `data.yaml` | Pad naar `data.yaml`. |
| `--naam` | `yolo` | Naam van de run; bepaalt `runs/<naam>/` en `weights/<naam>.pt`. **Gebruik per run een unieke naam**, anders wordt de vorige run overschreven. |
| `--model` | `yolov8n.pt` | Startmodel (`yolov8n/s/m/l/x.pt`); wordt automatisch gedownload. |
| `--epochs` | `100` | Aantal epochs. |
| `--imgsz` | `1280` | Beeldgrootte bij training. Gebruik dezelfde waarde bij testen. |
| `--batch` | `4` | Batchgrootte. Bij "CUDA out of memory": verlagen. |
| `--device` | `auto` | `cpu`, `0` (eerste GPU) of `auto`. |
| `--seed` | `0` | Seed voor herhaalbaarheid. |

**Wat het programma doet**

1. Traint het model en meet de totale trainingstijd.
2. Valideert `best.pt` op de validatieset en print mAP50, mAP50-95, precision en recall, algemeen en per klasse.
3. Schrijft die cijfers naar `runs/<naam>/metrics.txt`.
4. Kopieert `best.pt` naar `weights/<naam>.pt`.

**Voor een eerlijke CPU/GPU-vergelijking:** laat alles gelijk (dataset, model, epochs, imgsz, batch, seed) behalve
`--device` en `--naam`.

**Afbreken:** `Ctrl + C`. Ultralytics bewaart `last.pt` en `best.pt` in `runs/<naam>/weights/`, maar de mAP-meting en de
kopie naar `weights/` gebeuren pas na een volledige training. Hervatten kan met
`python -c "from ultralytics import YOLO; YOLO(r'runs\<naam>\weights\last.pt').train(resume=True)"`.

### 4.6 Programma 5 – YOLOv8 testen

```powershell
python 5_yolo_testen.py schemas_set1\Schema1.png schemas_set1\Schema2.png --gewichten weights\yolo_gpu.pt --imgsz 640
```

| Optie | Standaard | Betekenis |
|---|---|---|
| `--gewichten` | `weights/yolo.pt` | Getraind model (`.pt`). |
| `--conf` | `0.25` | Minimale zekerheid van het model. |
| `--iou` | `0.3` | NMS-drempel van YOLO zelf. |
| `--nested` | `0.6` | Extra filter: een box van dezelfde klasse die voor meer dan deze fractie in een betere box ligt, wordt verwijderd. `0` = uit. |
| `--imgsz` | `1280` | **Moet gelijk zijn aan de waarde bij het trainen.** |
| `--device` | `auto` | `cpu`, `0` of `auto`. |
| `--uit` | `resultaten` | Uitvoermap. |

De uitvoernamen bevatten de naam van het gewichtenbestand, zodat resultaten van verschillende modellen elkaar niet
overschrijven.

---

## 5. Uitvoerformaat

Elk programma schrijft per schema twee bestanden in `resultaten/`:

- `<programma>__<schema>.png`: het schema met gekleurde boxes en label + score. Elk label heeft een vaste kleur.
- `<programma>__<schema>.txt`: één regel per detectie, gescheiden door tabs:

  ```
  label    x    y    breedte    hoogte    score
  ```

  `x` en `y` zijn de linkerbovenhoek in pixels van het originele schema. De betekenis van de score verschilt per
  techniek:

  | Programma | Score |
  |---|---|
  | 1, 2 | Correlatiescore van de template matching (0–1). |
  | 3, 3b | Aantal inliers gedeeld door het aantal keypoints van het symbool (maximaal 1). |
  | 5 | Zekerheid (confidence) van het YOLO-model. |

  De scores van verschillende technieken zijn dus **niet rechtstreeks vergelijkbaar**.

---

## 6. Voorbeeldworkflow voor een experiment

1. Kies een symbolenmap (`symbolen_set1`) en testschema's (`schemas_set1\...`).
2. Draai programma 1, 2, 3 en 3b op dezelfde schema's, en programma 5 met een getraind model.
3. Vergelijk de afbeeldingen en txt-bestanden in `resultaten/`.
4. Zet een verbetering uit met een `--geen-...`-optie om het effect ervan afzonderlijk te bekijken, bijvoorbeeld:
   ```powershell
   python 2_template_matching_verbeterd.py schemas_set1\Schema1.png --geen-omvattend
   python 3b_feature_matching_verbeterd.py schemas_set1\Schema1.png --geen-groottes
   ```
5. Noteer per run de gebruikte opties en de versies van Python, `opencv-python` en `ultralytics`.

---

## 7. Veelvoorkomende problemen

| Probleem | Oorzaak / oplossing |
|---|---|
| `images not found, missing path ...` bij training | `path` in `data.yaml` klopt niet. Gebruik een absoluut pad naar de `dataset`-map. |
| `No module named torch` | Torch niet geïnstalleerd in de actieve venv. Activeer de venv en installeer PyTorch (stap 3 in hoofdstuk 3). |
| `torch.cuda.is_available()` geeft `False` | CPU-versie van PyTorch geïnstalleerd of geen geschikte driver. Installeer de CUDA-versie opnieuw. |
| `CUDA out of memory` | `--batch` of `--imgsz` verlagen. |
| YOLO vindt niets of verkeerde boxen bij testen | `--imgsz` bij testen verschilt van training, of `--conf` te hoog. |
| Template matching vindt niets | Symbool is veel groter dan in het schema (programma 1 zoekt enkel op originele grootte), of de drempel is te hoog. |
| Feature matching slaat een symbool over | Te weinig keypoints in het symbool (eenvoudige symbolen). Zie `--min-matches`. |
| Training overschreven | Dezelfde `--naam` gebruikt voor twee runs. Gebruik een unieke naam per run. |

---

## 8. Technische opmerkingen

- Op Windows draait de YOLO-training met meerdere workers; daarom staat de code van programma 4 in een
  `main()`-functie met `if __name__ == "__main__":`. Dit mag je niet verwijderen.
- Alle detecties worden teruggerekend naar de pixelcoördinaten van het originele schema, ook als intern is opgeschaald.
- Kleuren per label zijn afgeleid van de labelnaam (CRC32-hash), dus hetzelfde label heeft altijd dezelfde kleur.
- `.gitignore` sluit `venv/` en `__pycache__/` uit. Pas aan wat je wel of niet wil delen (bv. `runs/`, `weights/`,
  `resultaten/`).
