param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$libraryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $libraryRoot 'papers.csv'
$papers = Import-Csv -LiteralPath $manifestPath
$success = 0
$skipped = 0
$failed = [System.Collections.Generic.List[object]]::new()

function Test-PdfFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -lt 10000) { return $false }
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $bytes = New-Object byte[] 4
        $read = $stream.Read($bytes, 0, 4)
        if ($read -ne 4) { return $false }
        return ([System.Text.Encoding]::ASCII.GetString($bytes) -eq '%PDF')
    }
    finally {
        $stream.Dispose()
    }
}

foreach ($paper in $papers) {
    $destination = Join-Path $libraryRoot $paper.File
    $directory = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }

    if (-not $Force -and (Test-PdfFile -Path $destination)) {
        $skipped++
        Write-Output "SKIP  $($paper.Title)"
        continue
    }

    $temporary = "$destination.part"
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Force
    }

    try {
        & curl.exe --location --fail --silent --show-error --retry 3 --retry-delay 2 --connect-timeout 30 --max-time 240 --user-agent "Mozilla/5.0 GRAFT-research-library/1.0" --output $temporary $paper.PDF_URL
        if ($LASTEXITCODE -ne 0) {
            throw "curl exit code $LASTEXITCODE"
        }
        if (-not (Test-PdfFile -Path $temporary)) {
            throw "downloaded content is not a valid PDF"
        }
        Move-Item -LiteralPath $temporary -Destination $destination -Force
        $success++
        Write-Output "OK    $($paper.Title)"
    }
    catch {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
        $failed.Add([pscustomobject]@{
            Title = $paper.Title
            File = $paper.File
            URL = $paper.PDF_URL
            Error = $_.Exception.Message
        })
        Write-Output "FAIL  $($paper.Title) -- $($_.Exception.Message)"
    }
}

Write-Output ""
Write-Output "Downloaded: $success"
Write-Output "Already valid: $skipped"
Write-Output "Failed: $($failed.Count)"

if ($failed.Count -gt 0) {
    Write-Output ""
    Write-Output "Failed papers:"
    $failed | ForEach-Object { Write-Output "- $($_.Title) | $($_.URL) | $($_.Error)" }
    exit 2
}
