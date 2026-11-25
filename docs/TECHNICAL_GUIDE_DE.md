# Technische Vertiefung: Defektsegmentierung

## 1. Die konkrete Aufgabe

Eine Klassifikation beantwortet, ob ein Bild einen Fehler enthält. Eine Segmentierung bestimmt zusätzlich für jedes Pixel, ob es zum Fehler gehört. Der Unterschied ist für eine Sichtprüfung wesentlich: Eine einzige Bildklasse liefert weder Fehlerfläche noch Kontur oder Position. Dieses Projekt untersucht binäre Masken für kleine Oberflächenfehler.

Eingabe ist ein Graubild im Wertebereich 0 bis 1. Ausgabe ist zunächst ein Logit pro Pixel. Erst die Sigmoid-Funktion erzeugt eine Wahrscheinlichkeit; ein gewählter Schwellwert erzeugt daraus eine binäre Maske. Das Modell erhält beim Training Bilder und zugehörige Masken. Bei einer normalen Oberfläche besteht die Maske vollständig aus Nullen.

Der Versuch soll eine Architekturentscheidung überprüfen: Helfen lokale Attention und zusätzlicher Kontext gegenüber einer kleinen CNN-Basis? Deshalb werden beide Erweiterungen einzeln und gemeinsam untersucht. Ein reines Ergebnis der komplexesten Variante würde diese Frage nicht beantworten.

## 2. Den Datenfluss im Code verfolgen

Für eine Eingabe `[B, 1, 64, 64]` entstehen diese Formen:

| Stufe | Tensorform | Bedeutung |
|---|---|---|
| Stem | `[B, 16, 64, 64]` | Lokale Strukturen bei voller Auflösung |
| Encoder | `[B, 32, 32, 32]` | Stärkere Merkmale bei halber Auflösung |
| Bottleneck | `[B, 32, 16, 16]` | Größerer räumlicher Kontext |
| Attention | `[B, 32, 16, 16]` | Austausch innerhalb regulärer und verschobener Fenster |
| Fusion | `[B, 32, 32, 32]` | Verbindung feiner Merkmale mit gepooltem Kontext |
| Decoder-Eingabe | `[B, 48, 64, 64]` | 32 hochskalierte Kanäle plus 16 Stem-Kanäle |
| Kopf | `[B, 1, 64, 64]` | Ein Logit pro Pixel |

Ein Residualblock addiert seine Eingabe beziehungsweise eine 1×1-Projektion zu zwei Faltungsschritten. Der direkte Pfad erleichtert den Gradientenfluss. GroupNorm normalisiert Kanalgruppen pro Bild und benötigt keine geschätzten Batch-Statistiken. Das ist bei kleinen Batches nützlich; es beweist für diese Aufgabe noch keine Überlegenheit gegenüber anderen Normalisierungen.

Die Stem-Verbindung bewahrt feine Ortsinformation. Eine spätere Interpolation allein kann Details, die beim Downsampling verschwunden sind, nicht wiederherstellen. Für sehr kleine reale Fehler wären größere Eingaben oder ein sauber definierter Kachelansatz zu prüfen.

## 3. Q, K und V wirklich erklären

Ein 4×4-Fenster enthält 16 Tokens. Jeder Token besitzt 32 Merkmale. Eine lineare Projektion erzeugt Query, Key und Value. Bei vier Köpfen enthält jeder Kopf acht Merkmale pro Token.

Die Form der Q/K/V-Tensoren ist hier `[B × 16 Fenster, 4 Köpfe, 16 Tokens, 8 Merkmale]`. Das Skalarprodukt einer Query mit allen Keys beschreibt, welche Tokens relevante Informationen liefern können. Division durch `sqrt(8)` begrenzt die typische Größenordnung dieser Produkte. Eine Softmax über die Keys erzeugt Gewichte, deren Summe pro Query eins ist. Die gewichtete Summe der Values ist die neue Information für diesen Token.

Die Operation lautet:

$$A = \operatorname{softmax}(QK^T/\sqrt{d} + B + M),\qquad U=AV.$$

`B` ist ein lernbarer Bias für relative Positionen. Ein Fenster mit Seitenlänge vier hat relative Zeilen- und Spaltenabstände von −3 bis +3. Daraus entstehen 7×7 mögliche Abstandspaare. Der Code bildet jedes Paar auf einen Index ab und lernt pro Kopf einen Bias für diese 49 Positionen. Der Bias beschreibt relative Anordnung; er ist keine absolute Bildkoordinate.

`M` sperrt unzulässige Verbindungen. Der erste Block nutzt unverschobene Fenster. Der zweite verschiebt das Raster um zwei Pixel der Merkmalskarte. Dadurch können Informationen über bisherige Fenstergrenzen gelangen. Das zyklische Verschieben darf aber den linken mit dem rechten Bildrand nicht verbinden: Diese Nachbarschaft existiert in einem normalen Kamerabild nicht. Der Test `test_shift_does_not_connect_opposite_image_boundaries` verändert deshalb die letzte Spalte und überprüft die erste Spalte.

Nicht durch vier teilbare Merkmalskarten werden aufgefüllt. Die zusätzlichen Positionen dürfen für echte Queries keine gültigen Keys darstellen. Vollständig aufgefüllte Query-Zeilen erhalten einen endlichen Sperrwert, damit kein NaN durch eine Softmax über ausschließlich negative Unendlichkeiten entsteht. Ihre Ausgaben werden anschließend abgeschnitten. Tests prüfen endliche Gradienten auch bei rechteckigen und ungeraden Größen.

## 4. Warum Fenster statt globaler Attention?

Bei einer Merkmalskarte mit `H×W` Positionen besitzt globale Attention ungefähr `(HW)²` paarweise Beziehungen pro Kopf. Bei Fenstern mit Seitenlänge `k` entstehen ungefähr `HW×k²` Beziehungen. Für festes `k` wächst dieser Anteil linear mit der Zahl der Positionen. Die linearen Projektionen und das MLP verursachen zusätzlich Aufwand proportional zu `HW×C²`.

Fenster begrenzen allerdings den unmittelbaren Kontext. Verschiebungen verbinden benachbarte Fenster schrittweise; sie ersetzen keinen unbegrenzt globalen Informationsaustausch in einem einzelnen Block. Die gesamte Architektur bleibt außerdem von Kanalzahl, Speicherbewegungen, Faltungen und Laufzeitbibliothek abhängig. Aus der asymptotischen Attention-Komplexität folgt deshalb keine konkrete CPU-Latenz.

## 5. Cross-Scale-Fusion

Die feinere Karte hat im Demo 32×32 Positionen. Sie liefert die Queries. Die grobere Kontextkarte wird auf 4×4 Positionen gepoolt und liefert Keys und Values. Die Attention-Matrix enthält damit 1024×16 Einträge pro Bild. Eine vollständige globale Attention auf der 32×32-Karte hätte 1024×1024 Einträge pro Kopf.

Die Fusion soll die lokale Defektinterpretation durch Oberflächenkontext verbessern. Pooling kann jedoch kleine Fehler verwischen und Hintergrundmuster verstärken. Zusätzliche Parameter benötigen außerdem Trainingsinformation. Das gemessene Ergebnis zeigt, weshalb eine plausible Begründung experimentell überprüft werden muss.

## 6. Loss, Ungleichgewicht und leere Masken

Der überwiegende Teil eines Bildes ist Hintergrund. Ein Modell, das überall Hintergrund ausgibt, kann daher eine hohe Pixelgenauigkeit und trotzdem unbrauchbaren Defekt-Recall haben. Das Projekt kombiniert zwei Ziele.

Focal Loss gewichtet leicht klassifizierbare Pixel schwächer. Für die korrekte Klassenwahrscheinlichkeit `p_t` lautet sie `-alpha_t × (1-p_t)^gamma × log(p_t)`. Hier gelten `gamma=2`, positive Gewichtung 0,75 und negative Gewichtung 0,25. Die Berechnung nutzt numerisch stabile BCE mit Logits.

Soft Dice betrachtet die Überlappung von Wahrscheinlichkeit und Maske. Die Implementierung berechnet Dice zunächst pro Bild und mittelt danach. So dominiert nicht automatisch das Bild mit der größten Defektfläche die gesamte Batch-Maske. Ein Glättungsterm von eins hält vollständig leere Masken definiert. Die beiden Loss-Anteile werden gleich gewichtet.

Das ist eine konkrete Konfiguration, kein optimierter universeller Industriestandard. Loss-Gewichte, Auflösung und Auswahlkriterium müssen bei realen Daten auf unabhängigen Entwicklungsdaten begründet werden.

## 7. Auswahl und Auswertung sauber trennen

Trainingsbilder aktualisieren die Gewichte. Validierungsbilder wählen den besten Checkpoint und einen Schwellwert aus der vordefinierten Liste. Testbilder liefern die abschließenden Vergleichswerte. Eine Änderung anhand der Testfehler wäre ein neuer Entwicklungsschritt und benötigte für eine belastbare Abschlussbewertung neue unberührte Testdaten.

Die Kennzahlen verwenden echte Zählwerte:

$$\mathrm{IoU}=\frac{TP}{TP+FP+FN},\qquad \mathrm{Dice}=\frac{2TP}{2TP+FP+FN}.$$

Precision beantwortet, welcher Anteil vorhergesagter Fehlerpixel wirklich fehlerhaft ist. Recall beantwortet, welcher Anteil echter Fehlerpixel gefunden wird. Der Anteil normaler Bilder mit mindestens einem positiven Pixel ist eine andere Größe als der Anteil falsch positiver Pixel. Eine kleine falsch positive Fläche kann bereits einen ganzen Gutteil-Alarm auslösen.

Zusätzlich werden Testwerte nach synthetischem Fehlertyp und pro Bild gespeichert. Die Fehlerabbildungen zeigen die vier schlechtesten Dice-Fälle, nicht nur gut aussehende Beispiele. Das macht übersehene Kratzer, unvollständige Masken und Hintergrundalarme sichtbar.

## 8. Die gemessene Aussage richtig formulieren

Die CNN-Basis erreicht im Mittel etwa 0,605 Dice. Attention allein erreicht etwa 0,629. Die kombinierte Variante erreicht etwa 0,542. Die Streuung über drei Starts ist groß. Mit sechs Epochen und kleinem synthetischem Datensatz lässt sich daraus keine allgemeine Rangfolge von Architekturklassen ableiten.

Eine belastbare fachliche Aussage lautet: Unter dem aufgezeichneten Versuchsprotokoll liefert die kombinierte Erweiterung keinen stabilen Mehrwert. Das rechtfertigt, die einfachere Basis als Referenz beizubehalten und zuerst Datenmenge, Konvergenz und Parameterbudget zu untersuchen.

Die Seeds variieren Initialisierung und Reihenfolge; der Testdatensatz bleibt gleich. Drei Seeds sind deshalb keine drei unabhängigen Produktionsdatensätze. Weitere Versuche sollten auch andere Chargen, Materialien, Kameras und kleine Fehlergrößen berücksichtigen.

## 9. Fragen für ein technisches Review

**Warum keine fertige Detektionsbibliothek?** Eine Bounding Box löst die Maskenaufgabe nicht. Die explizite Implementierung ermöglicht außerdem isolierte Untersuchungen der hier gewählten Attention- und Fusionsbausteine.

**Warum reicht eine hohe Accuracy nicht?** Weil viele Hintergrundpixel die Genauigkeit dominieren können, obwohl das Modell fast alle Fehler übersieht. Dice, Recall und normale Bildalarme zeigen andere Eigenschaften.

**Warum ist das Ergebnis der komplexen Variante schwächer?** Der Versuch isoliert die beobachtete Wirkung, nicht ihre eindeutige Ursache. Wenige Daten, zusätzliche Parameter, kurze Trainingsdauer, Initialisierung und Informationsverlust durch Pooling sind plausible Hypothesen, die separate Kontrollen benötigen.

**Was ist noch nicht überprüft?** Reale Produktionsdaten, native Auflösung/Kacheln, ein auf Fehlalarmkosten abgestimmter Betriebspunkt, GPU/Edge-Laufzeit und exportierte Backend-Modelle. Die vorhandenen Checkpoints und CPU-Messungen decken diese Punkte nicht ab.
