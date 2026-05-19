$ErrorActionPreference = "Stop"

$command = ".\.venv\Scripts\python.exe manage.py test tests.test_source_pinning"
Write-Host "Running: $command"
Invoke-Expression $command
