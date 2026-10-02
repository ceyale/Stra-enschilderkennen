<#
kaggle/run.ps1 - Ein Skript fuer alles, was auf Kaggle laeuft.

Warum: Der Entwicklungsrechner kann nicht rechnen - die Windows-Anwendungssteuerung
(Smart App Control) blockiert die PyTorch-DLLs, siehe kaggle/README.md. Also laeuft das
Training auf Kaggle. Dieses Skript holt die Rohdaten, legt das Upload-Verzeichnis an,
schiebt Datensatz und Kernel hoch, fragt den Stand ab und holt die Ergebnisse ins Repo.

    .\kaggle\run.ps1 -Step raw       # fehlende Rohdaten holen (GTSRB-Test-GT, coco128)
    .\kaggle\run.ps1 -Step stage     # Upload-Verzeichnis anlegen (wird geprueft)
    .\kaggle\run.ps1 -Step probe     # Kernel-Befehle gegen das Staging durchspielen (ohne Netz)
    .\kaggle\run.ps1 -Step dataset   # Kaggle-Datensatz anlegen (einmalig) oder -NeueVersion
    .\kaggle\run.ps1 -Step push      # Kernel hochladen und starten
    .\kaggle\run.ps1 -Step status    # Stand abfragen (-Step logs zeigt das Protokoll)
    .\kaggle\run.ps1 -Step warten    # bis zum Ende warten, dann pull + install
    .\kaggle\run.ps1 -Step pull      # Ergebnisse herunterladen
    .\kaggle\run.ps1 -Step install   # Modelle + Fixture ins Repo kopieren
    .\kaggle\run.ps1 -Step alle      # raw -> stage -> probe -> dataset -> push

Wichtig: Der Kaggle-Datensatz enthaelt den Code (tools/) UND die Rohdaten, damit der
Kernel kein Netz braucht. Aendert sich tools/, muss der Datensatz neu hochgeladen werden
(-NeueVersion).
#>
param(
    [ValidateSet('alle', 'raw', 'stage', 'probe', 'dataset', 'push', 'status', 'logs',
                 'warten', 'pull', 'install', 'clean')]
    [string]$Step = 'alle',
    [string]$Staging = "$env:TEMP\schilder-kaggle",
    [string]$Out = 'data\kaggle-out',
    [switch]$Oeffentlich,        # Datensatz oeffentlich statt privat
    [switch]$NeueVersion,        # vorhandenen Datensatz aktualisieren statt neu anlegen
    [switch]$NurModelle,         # beim Abholen das grosse datensatz-ZIP auslassen
    [switch]$DatenErsetzen,      # data/det durch die auf Kaggle gebaute Fassung ersetzen
    [string]$Alt = 'models\signs-det.pt'   # Vergleichs-Checkpoint (der veroeffentlichte Stand)
)

$ErrorActionPreference = 'Stop'
# Kaggle-Protokolle enthalten Unicode; ohne diese Zeile bricht "kaggle kernels logs" mit
# "'charmap' codec can't encode characters" ab (gemessen).
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Repo = Split-Path -Parent $PSScriptRoot          # Skript liegt in kaggle/ -> eine Ebene hoch
$Stage = Join-Path $Staging 'schilderdet-raw'
$Kernel = 'raphbre/schilder-scanner-detektor-trainieren-gtsrb'
$Datensatz = 'raphbre/schilder-det-raw'
$Gtsrb = Join-Path $Repo 'data\gtsrb'
$Negativ = Join-Path $Repo 'data\negatives'
$Urteile = @{}                                    # Schritt -> Ergebnis, fuer die Schlusszeile


function Schritt($text) {
    Write-Host ''
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    Write-Host "[schritt] $text" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkGray
}


function Hol-Datei([string]$Url, [string]$Ziel, [long]$MindestGroesse, [string]$Was) {
    # Eine Datei nur holen, wenn sie fehlt oder offensichtlich abgebrochen ist.
    # Warum die Groessenpruefung: die GTSRB-Server liefern bei Abbruch eine kleine
    # Fehlerseite mit Status 200 - ohne Pruefung faellt das erst im Kernel auf.
    if ((Test-Path $Ziel) -and (Get-Item $Ziel).Length -ge $MindestGroesse) {
        Write-Host ("[ok]    {0} schon da: {1:N1} MB" -f (Split-Path $Ziel -Leaf), ((Get-Item $Ziel).Length / 1MB))
        return
    }
    New-Item -ItemType Directory -Force (Split-Path $Ziel) | Out-Null
    Write-Host "[hole]  $Was -> $Ziel"
    & curl.exe -L --fail --retry 3 --retry-delay 2 -o $Ziel $Url
    if ($LASTEXITCODE -ne 0) { throw "Download fehlgeschlagen (curl Code $LASTEXITCODE): $Url" }
    $groesse = (Get-Item $Ziel).Length
    if ($groesse -lt $MindestGroesse) {
        throw "$Was ist zu klein ($groesse Bytes, erwartet >= $MindestGroesse) - Abbruch statt Blindflug"
    }
    Write-Host ("[ok]    {0}: {1:N1} MB" -f (Split-Path $Ziel -Leaf), ($groesse / 1MB))
}


function Hol-Rohdaten {
    Schritt 'Rohdaten holen (GTSRB-Test-GT und coco128)'
    Hol-Datei 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip' `
        (Join-Path $Gtsrb 'gtsrb-test-gt.zip') 90000 'GTSRB Final Test GT (Labels des Test-Sets)'
    Hol-Datei 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip' `
        (Join-Path $Negativ 'coco128.zip') 6000000 'coco128 (128 echte Fotos, CC BY 4.0)'
    if (-not (Test-Path (Join-Path $Negativ 'coco128\images\train2017'))) {
        Write-Host '[entpacke] coco128 fuer die lokale Sichtpruefung'
        Expand-Archive -Path (Join-Path $Negativ 'coco128.zip') -DestinationPath $Negativ -Force
    }
    foreach ($name in 'gtsrb-train.zip', 'gtsrb-test.zip') {
        $pfad = Join-Path $Gtsrb $name
        if (-not (Test-Path $pfad)) { throw "es fehlt $pfad - siehe README.md (curl fuer GTSRB)" }
        Write-Host ("[ok]    {0}: {1:N1} MB" -f $name, ((Get-Item $pfad).Length / 1MB))
    }
    $Urteile['raw'] = 'Rohdaten vollstaendig'
}



function Verlinke([string]$Quelle, [string]$Ziel) {
    # Harte Verknuepfung statt Kopie: die 365 MB GTSRB-ZIPs liegen schon im Repo und
    # sollen nicht ein zweites Mal auf die Platte (und in die OneDrive-Sicherung).
    if (-not (Test-Path $Quelle)) { throw "Quelle fehlt: $Quelle" }
    try { New-Item -ItemType HardLink -Path $Ziel -Target $Quelle -Force | Out-Null }
    catch { Copy-Item $Quelle $Ziel }        # anderes Laufwerk: dann eben kopieren
    Write-Host ("[ok]    {0} ({1:N1} MB)" -f (Split-Path $Ziel -Leaf), ((Get-Item $Ziel).Length / 1MB))
}


function Leg-Staging {
    # Das Upload-Verzeichnis: Code + Rohdaten an einer Stelle, damit der Kernel kein Netz
    # braucht (enable_internet bleibt aus) und alle Quellen dieselben sind, die auch
    # data/gtsrb bzw. data/negatives lokal liegen.
    Schritt "Upload-Verzeichnis anlegen: $Stage"
    if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
    New-Item -ItemType Directory -Force $Stage | Out-Null
    Copy-Item (Join-Path $PSScriptRoot 'dataset-metadata.json') $Stage

    $toolsZiel = Join-Path $Stage 'tools'
    New-Item -ItemType Directory -Force $toolsZiel | Out-Null
    Get-ChildItem (Join-Path $Repo 'tools') -File -Filter '*.py' | Copy-Item -Destination $toolsZiel
    $toolsGroesse = (Get-ChildItem $toolsZiel | Measure-Object Length -Sum).Sum / 1KB
    Write-Host ("[ok]    tools/ ({0} Dateien, {1:N0} KB)" -f (Get-ChildItem $toolsZiel).Count, $toolsGroesse)

    New-Item -ItemType Directory -Force (Join-Path $Stage 'gtsrb'), `
        (Join-Path $Stage 'negatives'), (Join-Path $Stage 'user') | Out-Null
    foreach ($name in 'gtsrb-train.zip', 'gtsrb-test.zip', 'gtsrb-test-gt.zip') {
        Verlinke (Join-Path $Gtsrb $name) (Join-Path $Stage "gtsrb\$name")
    }
    Verlinke (Join-Path $Negativ 'coco128.zip') (Join-Path $Stage 'negatives\coco128.zip')
    Copy-Item (Join-Path $Repo 'Test\Nothing.jpg') (Join-Path $Stage 'user\Nothing.jpg')
    Write-Host ("[ok]    Nothing.jpg ({0:N0} KB)" -f `
        ((Get-Item (Join-Path $Stage 'user\Nothing.jpg')).Length / 1KB))

    # Vergleichs-Checkpoint: der bisher veroeffentlichte Stand wird im Kernel auf DERSELBEN
    # Messlatte mitgemessen. Noetig, weil die 200 synthetischen val-Bilder mit dem neuen
    # Tafel-Szenario viel mehr Boxen tragen - zwei Laeufe auf verschiedenen Messlatten lassen
    # sich nicht vergleichen.
    $altQuelle = Join-Path $Repo $Alt
    if (Test-Path $altQuelle) {
        New-Item -ItemType Directory -Force (Join-Path $Stage 'alt') | Out-Null
        Copy-Item $altQuelle (Join-Path $Stage 'alt\vergleich.pt') -Force
        Write-Host ("[ok]    Vergleichs-Checkpoint alt/vergleich.pt ({0:N1} MB) aus {1}" -f `
            ((Get-Item $altQuelle).Length / 1MB), $Alt)
    } else {
        Write-Host "[warnung] $altQuelle fehlt - der Vergleich alter/neuer Checkpoint entfaellt" -ForegroundColor Yellow
    }

    $gesamt = (Get-ChildItem $Stage -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
    Write-Host ("[fertig] {0}: {1:N0} MB in {2} Dateien" -f $Stage, $gesamt,
        (Get-ChildItem $Stage -Recurse -File).Count)
    $Urteile['stage'] = ("{0:N0} MB vorbereitet" -f $gesamt)
}


function Probe-Kernel {
    # Vor dem teuren Lauf pruefen, ob der Kernel die Wege findet und die richtigen Befehle
    # baut. Rechnet nichts und braucht kein Netz, faengt aber Pfadfehler ab, die sonst erst
    # nach dem Datensatzaufbau auffielen.
    Schritt 'Kernel-Befehle gegen das Upload-Verzeichnis pruefen'
    $pruef = Join-Path $Staging 'probe-work'
    if (Test-Path $pruef) { Remove-Item -Recurse -Force $pruef }
    New-Item -ItemType Directory -Force $pruef | Out-Null
    $env:SCHILDER_PROBE = '1'
    $env:SCHILDER_WORK = $pruef
    $env:SCHILDER_INPUT = $Staging
    try {
        & python (Join-Path $PSScriptRoot 'train_kernel.py')
        if ($LASTEXITCODE -ne 0) { throw "Probe des Kernels fehlgeschlagen (Code $LASTEXITCODE)" }
    } finally {
        Remove-Item Env:SCHILDER_PROBE, Env:SCHILDER_WORK, Env:SCHILDER_INPUT -ErrorAction SilentlyContinue
    }
    Write-Host "[ok]    Protokoll der Probe: $(Join-Path $pruef 'berichte\protokoll.txt')"
    $Urteile['probe'] = 'Wege und Befehle stimmen'
}



function Set-Datensatz {
    # Der Datensatz enthaelt Code UND Rohdaten. Achtung: "-r zip" ist Pflicht - mit "-r skip"
    # (Vorgabe) landen Unterordner NICHT im Datensatz (gemessen: nur Dateien der obersten
    # Ebene kamen an). Mit "-r zip" bleibt die Ordnerstruktur erhalten (gemessen: sub/b.txt).
    Schritt "Kaggle-Datensatz hochladen: $Datensatz"
    if (-not (Test-Path $Stage)) { throw "erst stage anlegen: .\kaggle\run.ps1 -Step stage" }
    $k = @('datasets', $(if ($NeueVersion) { 'version' } else { 'create' }), '-p', $Stage, '-r', 'zip')
    if ($NeueVersion) { $k += @('-m', "Stand $(Get-Date -Format 'yyyy-MM-dd HH:mm')") }
    if ($Oeffentlich -and -not $NeueVersion) { $k += '-u' }
    & kaggle @k
    if ($LASTEXITCODE -ne 0) { throw "kaggle $($k -join ' ') fehlgeschlagen (Code $LASTEXITCODE)" }
    & kaggle datasets status $Datensatz
    $Urteile['dataset'] = "hochgeladen: $Datensatz"
}


function Push-Kernel {
    Schritt "Kernel hochladen und starten: $Kernel"
    # -t ist die Obergrenze in Sekunden; 43 200 = 12 h (das Kaggle-Maximum). Der Lauf selbst
    # dauert rund zwei Stunden, die Grenze ist nur die Reissleine.
    & kaggle kernels push -p $PSScriptRoot -t 43200
    if ($LASTEXITCODE -ne 0) { throw "kaggle kernels push fehlgeschlagen (Code $LASTEXITCODE)" }
    & kaggle kernels status $Kernel
    $Urteile['push'] = "gestartet - Fortschritt: https://www.kaggle.com/code/$Kernel"
}


function Zeig-Status {
    Schritt "Stand abfragen: $Kernel"
    & kaggle kernels status $Kernel
    if ($LASTEXITCODE -ne 0) { throw "kaggle kernels status fehlgeschlagen (Code $LASTEXITCODE)" }
}


function Zeig-Logs {
    Schritt "Protokoll abholen: $Kernel"
    $log = Join-Path $Repo 'data\kaggle-log.txt'
    & kaggle kernels logs $Kernel | Tee-Object -FilePath $log
    Write-Host "[ok]    Protokoll zusaetzlich in $log"
}


function Hol-Ergebnis {
    Schritt "Ergebnisse herunterladen nach $Out"
    $ziel = Join-Path $Repo $Out
    New-Item -ItemType Directory -Force $ziel | Out-Null
    $k = @('kernels', 'output', $Kernel, '-p', $ziel, '-o')
    if ($NurModelle) {
        # Ohne das 400-MB-Archiv des Datensatzes. Das Muster greift auf die Dateinamen, nicht
        # auf die Ordner - deshalb Dateiendungen und nicht "models/".
        $k += @('--file-pattern', 'onnx|json|txt|\.pt$')
    }
    & kaggle @k
    if ($LASTEXITCODE -ne 0) { throw "kaggle kernels output fehlgeschlagen (Code $LASTEXITCODE)" }
    Get-ChildItem $ziel -Recurse -File | Select-Object FullName, Length | Format-Table -AutoSize
    $Urteile['pull'] = "in $ziel"
}



function Install-Ergebnis {
    # Modelle und Fixture dorthin kopieren, wo die App bzw. die JS-Tests sie erwarten.
    Schritt 'Ergebnisse ins Repo uebernehmen'
    $q = Join-Path $Repo $Out
    $bericht = Join-Path $q 'kaggle_report.json'
    if (-not (Test-Path $bericht)) { throw "$bericht fehlt - erst -Step pull" }

    Copy-Item (Join-Path $q 'models\*') (Join-Path $Repo 'models') -Force
    $namen = (Get-ChildItem (Join-Path $Repo 'models') -File | Select-Object -ExpandProperty Name) -join ', '
    Write-Host "[ok]    models/: $namen"
    $fixture = Join-Path $q 'berichte\model-out.json'
    if (Test-Path $fixture) {
        New-Item -ItemType Directory -Force (Join-Path $Repo 'tests\fixtures') | Out-Null
        Copy-Item $fixture (Join-Path $Repo 'tests\fixtures\model-out.json') -Force
        Write-Host '[ok]    tests/fixtures/model-out.json (Gegenprobe Modell <-> src/model.js)'
    }

    # Die Zahlen aus der Kurzfassung zeigen - sonst muesste man fuenf Dateien oeffnen.
    $r = Get-Content $bericht -Raw | ConvertFrom-Json
    $minuten = ($r.dauer_min.PSObject.Properties | Measure-Object -Property Value -Sum).Sum
    Write-Host ''
    Write-Host ("  Geraet: {0}   Gesamtdauer: {1:N0} min" -f $r.geraet, $minuten)
    foreach ($split in 'val.json', 'neg.json', 'val384.json') {
        $m = $r.kennzahlen.$split
        if ($m) {
            Write-Host ("  {0,-11} P={1:N3}  R={2:N3}  F1={3:N3}   tp={4} fp={5} fn={6}" -f `
                $split.Replace('.json', ''), $m.precision, $m.recall, $m.f1, $m.tp, $m.fp, $m.fn)
        }
    }
    $Urteile['install'] = 'models/ und Fixture aktualisiert'

    if ($DatenErsetzen) {
        $zip = Join-Path $q 'data\det.zip'
        if (-not (Test-Path $zip)) { throw "$zip fehlt (ohne -NurModelle abholen)" }
        Write-Host ''
        Write-Host '[achtung] data/det wird durch die auf Kaggle gebaute Fassung ersetzt' -ForegroundColor Yellow
        if (Test-Path (Join-Path $Repo 'data\det')) { Remove-Item -Recurse -Force (Join-Path $Repo 'data\det') }
        # Im Archiv liegen die Wege relativ zum Arbeitsverzeichnis (data/det/...) - deshalb
        # wird ins Repo-Wurzelverzeichnis entpackt, data/det entsteht dabei von allein.
        Expand-Archive -Path $zip -DestinationPath $Repo -Force
        $bilder = (Get-ChildItem (Join-Path $Repo 'data\det\images') -File).Count
        Write-Host "[ok]    data/det: $bilder Bilder"
    }
}


function Warte-Auf-Lauf {
    # Bis zum Ende warten, dann Ergebnisse holen und uebernehmen - ein Befehl statt drei.
    # Sinnvoll, weil ein Lauf rund zwei Stunden braucht und die API das Protokoll erst
    # danach herausgibt: der Status ist waehrend des Laufs das einzige Lebenszeichen.
    Schritt "Warten bis zum Ende des Laufs: $Kernel"
    while ($true) {
        $status = (& kaggle kernels status $Kernel 2>&1) -join ' '
        Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $status.Trim())
        if ($status -match 'COMPLETE') { break }
        if ($status -match 'ERROR|CANCEL') {
            Write-Host "[fehler] Lauf nicht erfolgreich - Protokoll mit -Step logs" -ForegroundColor Red
            throw "Lauf endete mit: $($status.Trim())"
        }
        Start-Sleep -Seconds 60
    }
    Hol-Ergebnis
    Install-Ergebnis
    $Urteile['warten'] = 'Lauf fertig, Ergebnisse uebernommen'
}


function Aufraeumen {
    Schritt 'Aufraeumen'
    if (Test-Path $Staging) {
        Remove-Item -Recurse -Force $Staging
        Write-Host "[ok]    $Staging entfernt"
    }
    Write-Host '[hinweis] Datensatz und Kernel bleiben online - zum Loeschen:' -ForegroundColor DarkGray
    Write-Host "          kaggle datasets delete $Datensatz -y" -ForegroundColor DarkGray
    Write-Host "          kaggle kernels delete $Kernel -y" -ForegroundColor DarkGray
}



# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------
switch ($Step) {
    'raw'     { Hol-Rohdaten }
    'stage'   { Leg-Staging }
    'probe'   { if (-not (Test-Path $Stage)) { Leg-Staging }; Probe-Kernel }
    'dataset' { if (-not (Test-Path $Stage)) { Leg-Staging }; Set-Datensatz }
    'push'    { Push-Kernel }
    'status'  { Zeig-Status }
    'logs'    { Zeig-Logs }
    'warten'  { Warte-Auf-Lauf }
    'pull'    { Hol-Ergebnis }
    'install' { Install-Ergebnis }
    'clean'   { Aufraeumen }
    'alle'    { Hol-Rohdaten; Leg-Staging; Probe-Kernel; Set-Datensatz; Push-Kernel }
}

if ($Urteile.Count -gt 0) {
    Write-Host ''
    Write-Host '[zusammenfassung]' -ForegroundColor Cyan
    foreach ($e in $Urteile.GetEnumerator()) { Write-Host ("  {0,-8} {1}" -f $e.Key, $e.Value) }
    if ($Step -eq 'push' -or $Step -eq 'alle') {
        Write-Host ''
        Write-Host 'Naechster Schritt:' -ForegroundColor Cyan
        Write-Host '  .\kaggle\run.ps1 -Step status     # laeuft es? (-Step logs zeigt das Protokoll)'
        Write-Host '  .\kaggle\run.ps1 -Step pull       # wenn fertig: Ergebnisse holen'
        Write-Host '  .\kaggle\run.ps1 -Step install    # Modelle + Fixture uebernehmen'
    }
}
