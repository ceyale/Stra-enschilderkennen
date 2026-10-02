# Architektur, Latenz und Qualitätsstudien

Die dazugehörige [Architekturübersicht als SVG](ARCHITEKTUR-UND-EXPERIMENTE.svg) zeigt Datenfluss, Skalen, Kopf-Ausgaben und den Weg bis in den Browser. Dieses Dokument ergänzt sie um einen reproduzierbaren Plan für die nächsten Studien.

## Stand und Grenzen der vorhandenen Evidenz

Der vorherige Kaggle-Lauf hat 220 Epochen trainiert und am Ende bei 320 px auf dem Projekt-Validierungssplit P=0,932, R=0,851 und F1=0,890 erreicht. Das Modell wurde als Checkpoint gespeichert; der Kernel scheiterte erst beim ONNX-Export an der fehlenden Bibliothek `onnxruntime`. Der laufende Wiederholungsversuch installiert die Abhängigkeit vor dem Training und erhöht die Stichproben je Epoche von 1.600 auf 2.400. Seine Endmetriken sind noch offen.

Diese Kennzahl ist **kein unabhängiger Straßentest**. Die Validierung besteht überwiegend aus GTSRB-Testausschnitten, die zu Szenen komponiert wurden, plus synthetischen Beispielen für zwei fehlende Typen. Der Negativsplit misst Hintergrundbilder, ist aber klein und enthält teilweise Bildquellen, die auch im Training vorkommen. Ergebnisse darauf sagen daher nicht zuverlässig voraus, wie das Modell auf einer neuen Dashcam, bei Nacht oder auf echten großen Szenen funktioniert.

## Verbesserungen, nach erwarteter Wirkung sortiert

1. **Echte, unabhängige Straßenszenen sammeln.** Der größte bekannte Engpass ist der Domänenabstand zwischen komponierten GTSRB-Ausschnitten und Schildern in der Umgebung. Daten nach Fahrt, Kamera oder Ort gruppiert aufteilen, damit fast identische Frames nicht über Training und Validierung verteilt werden. Personen- und Kennzeichenbereiche vor Speicherung unkenntlich machen.
2. **Kleine Schilder getrennt messen.** Recall nach Box-Diagonale und Bildposition ausweisen. Die stride-4-Stufe und 384-px-Eingabe sollen kleine Objekte helfen; ihr Effekt muss auf echten Szenen bestätigt werden.
3. **Fehlalarme gezielt nachtrainieren.** Falsch-positive Boxen aus einem separaten Pool sammeln, prüfen und als leere Bilder oder korrekt beschriftete Szenen ergänzen. Den späteren Testpool dabei unangetastet lassen.
4. **Auflösung gegen Latenz abwägen.** Die App bietet 256/320/384/448 px an. Frühere Werte 0,796/0,837/0,854 F1 stammten von einem anderen Checkpoint und gelten nicht als Resultat des aktuellen Modells. Jede Eingabegröße neu messen.
5. **Modelleffizienz separat ablatieren.** `fast`, `balanced` und `quality` ändern Breite, Tiefe bzw. Kontextblöcke. Nur derselbe Datensatz, dieselbe Trainingsstichprobe und derselbe Auswertungsweg erlauben einen fairen Vergleich. Weniger FLOPs versprechen keine proportionale Browser-Latenz.

## Experimentmatrix

| Studie | Änderung | Konstant halten | Hauptergebnis | Erwartung / Risiko |
|---|---|---|---|---|
| A: Stichprobenbudget | 100 vs. 150 Schritte/Epoche | Seed, Architektur, Bilder, 320 px | F1, mAP50, Lernkurve | Mehr Datenabdeckung kann helfen; Übertraining und Mehrkosten beobachten |
| B: Modellgröße | `fast`, `balanced`, `quality` | Auflösung, Daten, Optimierer, Updatezahl | Qualität und p50/p95-Latenz | `fast` dürfte weniger Rechenlast haben; möglicher Recallverlust |
| C: Eingabeauflösung | 256, 320, 384, 448 px | Checkpoint und Bildliste | Recall kleiner Schilder, F1, Latenz | Hohe Auflösung hilft kleinen Objekten; ungefähr quadratische Kostensteigerung |
| D: Hard negatives | Mining an/aus | Checkpoint, positive Beispiele | FP/Bild auf neuem Negativpool | Fehlalarme sollten sinken; Recall auf seltenen Klassen darf nicht einbrechen |
| E: reale Szenen | nur GTSRB-Kompositionen vs. ergänzt | Architektur und Updatezahl | Szene-gehaltener Testsatz | Wahrscheinlich stärkster Transfergewinn; Annotation ist teuer |
| F: Augmentation | Zoom/Degradation einzeln variieren | gleiche Seeds und Updates | kleine-/große-Objektmetriken | Zu starke Synthese kann Bildstatistik verfälschen |

### Versuchsregeln

- Mindestens drei Seeds je Konfiguration; Mittelwert und Standardabweichung berichten. Für ein schnelles Screening ist ein Seed erlaubt, aber nicht für eine abschließende Modellwahl.
- Trainings-, Tuning- und unabhängiger Testsplit nach **Originalszene/Fahrt** trennen, nicht nach abgeleiteten Einzelbildern. Kein Bild des abschließenden Testsplits zum Hard-negative-Mining verwenden.
- Neben F1/mAP50 pro Klasse und Objektgröße auch `FP/Bild` auf Negativbildern, Recall, Fehlklassifikationen und Konfidenzkalibrierung ausweisen. Die Konfidenzschwelle auf einem Tuning-Split festlegen.
- Browser-Latenz auf den Zielgeräten mit warmgeladener Session messen: p50/p95 für Canvas-Vorbereitung, ONNX-Lauf, Decode/NMS und Gesamtzeit. Je Gerät WebGPU und WASM getrennt erfassen; Energieverbrauch und thermisches Drosseln bei längeren Videoläufen mitnotieren.
- Einen Pareto-Vergleich führen: Ein Modell ist nur dann klar besser, wenn es bei ähnlicher Latenz mehr Qualität liefert oder bei ähnlicher Qualität schneller ist. `fast` nicht allein anhand von Parameterzahl oder FLOPs auswählen.

## Priorisierte nächste Runde

1. Den aktuellen Kaggle-Retry vollständig exportieren und dessen Qualität sowie Negativ-FP prüfen.
2. Den 150-Schritte-Lauf mit dem früheren 100-Schritte-Lauf vergleichen. Da beide denselben Seed und dieselbe Datenpipeline verwenden, ist das ein nützlicher erster Hinweis, aber noch keine Mehrseed-Studie.
3. Den neuen Checkpoint auf 256/320/384/448 px und in Browser-WASM/WebGPU messen; so erhält man zuerst einen Latency-Qualitäts-Kompromiss ohne erneutes Training.
4. Erst danach `fast` und `quality` als kontrollierte Ablation trainieren.
5. Für eine belastbare Feldentscheidung einen kleinen, unabhängigen und nach Szene gruppierten Satz deutscher Straßenschilder annotieren. Das ist aussagekräftiger als immer größere synthetische Splits.

## Latenzoption in der App

Die Größenwahl ist schon vorhanden: 256 px ist als schneller Modus beschriftet, 320 px ist der automatische Videowert und Fotos laufen automatisch mit 384 px. Die Modellgewichte bleiben gleich; die Rechenlast ändert sich hauptsächlich mit der Fläche des Eingangs. Ob 256 px auf dem aktuellen Retry die Qualitätsanforderung erfüllt, muss auf demselben Checkpoint gemessen werden.
