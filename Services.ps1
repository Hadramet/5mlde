function Check-Dependencies {
  if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed. Please install Docker and try again."
    exit 1
  }

  if (-not (Get-Command "docker-compose" -ErrorAction SilentlyContinue)) {
    Write-Host "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
  }
}

function Start-Services {
  Check-Dependencies

  $workers = Read-Host "Enter the number of workers (default: $($env:NUM_WORKERS -as [int])})"
  if ([string]::IsNullOrWhiteSpace($workers)) {
    $workers = $env:NUM_WORKERS -as [int]
  }

  Write-Host "Starting services with $workers worker(s)..."
  Set-Location src
  docker-compose up --scale worker=$workers -d
}

function Stop-Services {
  Check-Dependencies

  Write-Host "Stopping services and removing volumes..."
  Set-Location src
  docker-compose down --volumes
}

if ($args[0] -eq "start-services") {
  Start-Services
} elseif ($args[0] -eq "stop-services") {
  Stop-Services
} else {
  Write-Host "Usage: .\$($MyInvocation.MyCommand.Name) <start-services|stop-services>"
}
