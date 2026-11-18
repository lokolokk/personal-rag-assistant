<#
.SYNOPSIS
  Back up the cloud data, list the backups, or restore the latest one onto this laptop.

.DESCRIPTION
  Backup   Asks the running cloud server to save a cold copy of its data to the private S3 bucket.
  List     Shows the saved backups.
  Restore  Downloads the newest backup, checks its checksum and (only with -Apply) loads it into the
           laptop volume. Without -Apply it is a dry run: nothing on this laptop changes.

  Needs: AWS CLI, an AWS login (aws login --profile rag-aws), Docker Desktop for Restore.
  Never prints data: only names, sizes and counts.

.EXAMPLE
  .\scripts\cloud_data.ps1 -Action Backup
  .\scripts\cloud_data.ps1 -Action Restore            # dry run
  .\scripts\cloud_data.ps1 -Action Restore -Apply -CloudStopped
#>
param(
  [Parameter(Mandatory)][ValidateSet('Backup', 'List', 'Restore')][string]$Action,
  [switch]$Apply,
  [switch]$CloudStopped,
  [string]$AwsProfile = 'rag-aws',
  [string]$InstanceId = 'i-0f775d127492d3525',
  [string]$Region = 'ap-southeast-2',
  [string]$Volume = 'raglive_live-storage'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:AWS_REGION = $Region

$aws = (Get-Command aws -ErrorAction SilentlyContinue).Source
if (-not $aws) { $aws = 'C:\Program Files\Amazon\AWSCLIV2\aws.exe' }
& $aws configure export-credentials --profile $AwsProfile --format powershell | Invoke-Expression
if (-not $env:AWS_ACCESS_KEY_ID) { throw "No AWS session. Run: aws login --profile $AwsProfile" }

function Get-Bucket {
  $b = & $aws ssm get-parameter --name /rag/data_bucket --query Parameter.Value --output text
  if (-not $b) { throw 'Could not read /rag/data_bucket.' }
  return $b
}

if ($Action -eq 'Backup') {
  $ping = & $aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$InstanceId" --query 'InstanceInformationList[0].PingStatus' --output text
  if ($ping -ne 'Online') { throw 'The cloud server is not running. Run the "Start demo" workflow first.' }
  $params = New-TemporaryFile
  @{ commands = @('/opt/rag/bin/rag-backup.sh 2>&1 | tail -n 15') } | ConvertTo-Json | Set-Content $params -Encoding ascii
  $id = & $aws ssm send-command --instance-ids $InstanceId --document-name AWS-RunShellScript --comment 'backup now' --parameters "file://$params" --query Command.CommandId --output text
  Remove-Item $params -Force
  $st = ''
  for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 5
    $st = & $aws ssm get-command-invocation --command-id $id --instance-id $InstanceId --query Status --output text
    if ($st -in 'Success', 'Failed', 'TimedOut', 'Cancelled') { break }
  }
  "status=$st"
  & $aws ssm get-command-invocation --command-id $id --instance-id $InstanceId --query '[StandardOutputContent,StandardErrorContent]' --output text
  if ($st -ne 'Success') { exit 1 }
  return
}

$bucket = Get-Bucket

if ($Action -eq 'List') {
  & $aws s3 ls "s3://$bucket/backups/" | Select-String '\.tgz$'
  return
}

# ---- Restore ----
$keys = & $aws s3api list-objects-v2 --bucket $bucket --prefix backups/ --query "Contents[?ends_with(Key,'.tgz')].Key" --output json | ConvertFrom-Json
if (-not $keys) { throw 'No backups found in the bucket.' }
$key = @($keys | Sort-Object)[-1]   # @() keeps a single backup a list, not a string
$file = Split-Path $key -Leaf
$work = Join-Path (Split-Path $PSScriptRoot) 'tmp\cloud-restore'   # tmp/ is git-ignored
New-Item -ItemType Directory -Force -Path $work | Out-Null

& $aws s3 cp "s3://$bucket/$key" "$work\$file" --only-show-errors
& $aws s3 cp "s3://$bucket/$key.sha256" "$work\$file.sha256" --only-show-errors
$expected = ((Get-Content "$work\$file.sha256" -Raw).Trim() -split '\s+')[0].ToLower()
$actual = (Get-FileHash "$work\$file" -Algorithm SHA256).Hash.ToLower()
if ($expected -ne $actual) { throw "Checksum mismatch for $file. Download deleted nothing; restore stopped." }
"Latest backup: $file ($((Get-Item "$work\$file").Length) bytes), checksum OK."

if (-not $Apply) {
  'Dry run only. Nothing on this laptop changed.'
  "To load it into volume '$Volume': stop the cloud server first, then run with -Apply -CloudStopped."
  return
}

# Guards: one ngrok agent per domain, and nothing may be using the volume.
if (-not $CloudStopped) { throw 'Add -CloudStopped to confirm the cloud server (and its tunnel) is stopped.' }
if (Get-Process | Where-Object { $_.Name -like '*ngrok*' }) { throw 'An ngrok process is running on this laptop. Stop it first.' }
if (docker ps -q --filter "volume=$Volume") { throw "A running container uses volume '$Volume'. Stop it first." }
$image = docker images --format '{{.Repository}}:{{.Tag}}' | Select-String 'rag-backend|raglive-backend' | Select-Object -First 1
if (-not $image) { throw 'No local backend image to run tar with.' }
$image = $image.ToString()

# Safety copy of what is on the laptop now (read-only, non-root, no network).
$stamp = Get-Date -Format 'yyyyMMddTHHmmss'
docker run --rm --network none --user 10001:10001 --cap-drop ALL -v "${Volume}:/data:ro" -v "${work}:/out" --entrypoint tar $image --numeric-owner -czf "/out/pre-restore-$stamp.tgz" -C /data .
"Safety copy of the current laptop data: tmp\cloud-restore\pre-restore-$stamp.tgz"

# Scripts are mounted as files to avoid shell-quoting problems.
[IO.File]::WriteAllText("$work\restore.sh", "set -e`nfind /data -mindepth 1 -delete`ntar -xzf `"/in/`$1`" -C /data --numeric-owner`nchown -R 10001:10001 /data`n", (New-Object Text.UTF8Encoding $false))
[IO.File]::WriteAllText("$work\count.sh", "set -e`necho archive=`$(tar -tzf `"/in/`$1`" | grep -vc '/`$' || true)`necho volume=`$(find /data -type f | wc -l)`n", (New-Object Text.UTF8Encoding $false))
docker run --rm -u 0 --network none -v "${Volume}:/data" -v "${work}:/in:ro" --entrypoint sh $image /in/restore.sh $file
docker run --rm --network none --user 10001:10001 --cap-drop ALL -v "${Volume}:/data:ro" -v "${work}:/in:ro" --entrypoint sh $image /in/count.sh $file
"Restore finished. If archive and volume counts match, start the laptop stack as usual (Start RAG)."
