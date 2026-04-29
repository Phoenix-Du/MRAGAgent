$ErrorActionPreference = "Stop"

function Test-HttpHealth {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Url
    )

    try {
        $req = [System.Net.WebRequest]::Create($Url)
        $req.Method = "GET"
        $req.Timeout = 6000
        $req.ReadWriteTimeout = 6000
        $resp = $req.GetResponse()
        $statusCode = [int]([System.Net.HttpWebResponse]$resp).StatusCode
        if ($statusCode -ge 200 -and $statusCode -lt 300) {
            return [PSCustomObject]@{
                Service = $Name
                Url = $Url
                Status = "OK"
                HttpCode = $statusCode
                Message = "healthy"
            }
        }
        return [PSCustomObject]@{
            Service = $Name
            Url = $Url
            Status = "FAIL"
            HttpCode = $statusCode
            Message = "unexpected status code"
        }
    } catch {
        return [PSCustomObject]@{
            Service = $Name
            Url = $Url
            Status = "FAIL"
            HttpCode = "-"
            Message = $_.Exception.Message
        }
    } finally {
        if ($resp) {
            $resp.Close()
        }
    }
}

$checks = @(
    @{ Name = "orchestrator"; Url = "http://127.0.0.1:8000/healthz" },
    @{ Name = "rasa"; Url = "http://127.0.0.1:5005/version" },
    @{ Name = "raganything-bridge"; Url = "http://127.0.0.1:9002/healthz" },
    @{ Name = "image-pipeline"; Url = "http://127.0.0.1:9010/healthz" }
)

$results = @()
foreach ($item in $checks) {
    $results += Test-HttpHealth -Name $item.Name -Url $item.Url
}

$results | Format-Table -AutoSize

$failed = @($results | Where-Object { $_.Status -eq "FAIL" })
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Some services are not healthy. Start missing services with scripts in ./scripts." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "All services are healthy." -ForegroundColor Green
exit 0
