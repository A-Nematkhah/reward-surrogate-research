# Run one Phase-4 PPO pilot job. Prefers Docker; falls back to local Python.
param(
    [string]$RunId = "phase4_2d_v1",
    [int]$PilotTimesteps = 0
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($docker) {
    Write-Host "Using Docker isolation"
    docker compose -f docker-compose.triage.yml run --rm triage-labeler `
        python scripts/label_phase4_job.py --run-id $RunId --max-jobs 1
    exit $LASTEXITCODE
}

Write-Warning "docker not found; running local pilot (weaker isolation)"
if ($PilotTimesteps -gt 0) {
    $env:RSR_PILOT_TIMESTEPS = "$PilotTimesteps"
}
python scripts/label_phase4_job.py --run-id $RunId --max-jobs 1 --seed 0
exit $LASTEXITCODE
