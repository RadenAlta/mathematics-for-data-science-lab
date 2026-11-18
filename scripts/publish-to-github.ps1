# Run this from the extracted repository root, using PowerShell 7+.
# Requires Git and GitHub CLI (gh). Requires an interactive authenticated account.
# Example: pwsh -File .\scripts\publish-to-github.ps1
$ErrorActionPreference = 'Stop'
$repo = 'RadenAlta/mathematics-for-data-science-lab'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $root

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git is not installed. Install Git for Windows first.'
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'GitHub CLI (gh) is not installed. See https://cli.github.com/'
}
& gh auth status
if ($LASTEXITCODE -ne 0) {
    throw 'Log in first using: gh auth login'
}

& gh repo view $repo *> $null
if ($LASTEXITCODE -eq 0) {
    throw "The repo $repo already exists. This script will not overwrite an existing repository."
}
if (-not (Test-Path '.git')) {
    & git init -b main
    if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
}
& git add README.md README.id.md index.html src tests notebooks assets reports scripts requirements.txt .gitignore .github
if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
& git diff --cached --stat
& git commit -m 'docs: publish reproducible data science portfolio'
if ($LASTEXITCODE -ne 0) { throw 'git commit failed. Configure git user.name and user.email, or inspect the repository state.' }
& gh repo create $repo --public --source . --remote origin --push --description 'Reproducible Python portfolio: Naive Bayes, statistics and A/B testing'
if ($LASTEXITCODE -ne 0) { throw 'GitHub repo creation or push failed.' }
Write-Host "Published: https://github.com/$repo" -ForegroundColor Green
