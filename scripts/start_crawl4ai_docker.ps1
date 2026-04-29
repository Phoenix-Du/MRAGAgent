$ErrorActionPreference = "Stop"

$dockerExe = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
if (!(Test-Path $dockerExe)) {
    $dockerExe = "docker"
}

& $dockerExe pull unclecode/crawl4ai:0.8.6

$existing = & $dockerExe ps -a --filter "name=^crawl4ai$" --format "{{.Names}}"
if ($existing -eq "crawl4ai") {
    & $dockerExe rm -f crawl4ai | Out-Null
}

& $dockerExe run -d `
  -p 11235:11235 `
  --name crawl4ai `
  --shm-size=1g `
  unclecode/crawl4ai:0.8.6 | Out-Null

Write-Host "crawl4ai started at http://127.0.0.1:11235"
