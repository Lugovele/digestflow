$python = ".\.venv\Scripts\python.exe"
$manage = "manage.py"

if (-not (Test-Path $python)) {
    Write-Error "Python interpreter not found at $python"
    exit 1
}

$command = @($manage, "test", "tests.test_source_candidates")
Write-Host ("Running: {0} {1}" -f $python, ($command -join " "))

& $python @command
exit $LASTEXITCODE
