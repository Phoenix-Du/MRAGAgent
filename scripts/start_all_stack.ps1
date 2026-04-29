$ErrorActionPreference = "Stop"

function Start-InBackground {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$ScriptPath
    )

    $fullPath = Resolve-Path $ScriptPath
    Start-Process powershell -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $fullPath
    ) | Out-Null
    Write-Host "Started $Name via $ScriptPath"
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# 1) rasa parse api
Start-InBackground -Name "rasa" -ScriptPath ".\scripts\start_rasa_parse.ps1"
Start-Sleep -Seconds 3

# 2) raganything bridge api
Start-InBackground -Name "raganything-bridge" -ScriptPath ".\scripts\start_raganything_bridge.ps1"
Start-Sleep -Seconds 3

# 3) image pipeline api
Start-InBackground -Name "image-pipeline" -ScriptPath ".\scripts\start_image_pipeline.ps1"

Write-Host ""
Write-Host "Waiting 10 seconds before health check..."
Start-Sleep -Seconds 10

powershell -ExecutionPolicy Bypass -File ".\scripts\check_stack_health.ps1"
