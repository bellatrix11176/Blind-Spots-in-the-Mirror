# ============================================================
# organize_project.ps1
# Moves raw dataset files and task code into correct locations
# Run from: C:\Users\gigih\OneDrive\Current Projects\Kaggle Hackathon
# ============================================================

$root = "C:\Users\gigih\OneDrive\Current Projects\Kaggle Hackathon"

# ── Create folder structure if it doesn't exist ──────────────
$folders = @(
    "$root\data\raw",
    "$root\data\processed",
    "$root\src\tasks\task1_knowledge",
    "$root\src\tasks\task2_monitoring",
    "$root\src\tasks\task3_control",
    "$root\src\utils",
    "$root\src\analysis",
    "$root\output\results",
    "$root\output\reports"
)

foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
        Write-Host "Created: $folder" -ForegroundColor Green
    } else {
        Write-Host "Exists:  $folder" -ForegroundColor Gray
    }
}

# ── Move raw dataset files ────────────────────────────────────
$datasetMoves = @(
    @{ From = "$root\task1_knowledge_raw.jsonl"; To = "$root\data\raw\task1_knowledge_raw.jsonl" },
    @{ From = "$root\task2_monitoring_raw.jsonl"; To = "$root\data\raw\task2_monitoring_raw.jsonl" },
    @{ From = "$root\task3_control_raw.jsonl";   To = "$root\data\raw\task3_control_raw.jsonl" }
)

Write-Host "`n-- Moving dataset files --" -ForegroundColor Cyan
foreach ($move in $datasetMoves) {
    if (Test-Path $move.From) {
        Move-Item -Path $move.From -Destination $move.To -Force
        Write-Host "Moved: $($move.From | Split-Path -Leaf) --> data\raw\" -ForegroundColor Green
    } else {
        Write-Host "Not found: $($move.From | Split-Path -Leaf)" -ForegroundColor Yellow
    }
}

# ── Move task code files (if they are in root) ───────────────
$codeMoves = @(
    @{ From = "$root\task1_knowledge.py"; To = "$root\src\tasks\task1_knowledge\task1_knowledge.py" },
    @{ From = "$root\task2_monitoring.py"; To = "$root\src\tasks\task2_monitoring\task2_monitoring.py" },
    @{ From = "$root\task3_control.py";   To = "$root\src\tasks\task3_control\task3_control.py" }
)

Write-Host "`n-- Moving task code files --" -ForegroundColor Cyan
foreach ($move in $codeMoves) {
    if (Test-Path $move.From) {
        Move-Item -Path $move.From -Destination $move.To -Force
        $taskFolder = $move.To | Split-Path -Parent | Split-Path -Leaf
        Write-Host "Moved: $($move.From | Split-Path -Leaf) --> src\tasks\$taskFolder\" -ForegroundColor Green
    } else {
        Write-Host "Not found (may already be placed): $($move.From | Split-Path -Leaf)" -ForegroundColor Yellow
    }
}

# ── Final structure report ────────────────────────────────────
Write-Host "`n-- Final structure --" -ForegroundColor Cyan
Get-ChildItem -Path $root -Recurse -File |
    Where-Object { $_.FullName -notlike "*kaggle-benchmarks-ci*" } |
    ForEach-Object {
        $relative = $_.FullName.Replace($root + "\", "")
        Write-Host "  $relative"
    }

Write-Host "`nDone." -ForegroundColor Green
