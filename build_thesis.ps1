$OutputDirectory = "Report"

# Ensure the output directory exists
if (!(Test-Path $OutputDirectory)) {
    New-Item -ItemType Directory -Path $OutputDirectory
}

Write-Host "Building Thesis.tex into the $OutputDirectory folder..." -ForegroundColor Cyan

# Run latexmk with output redirection
# -pdf: generate pdf
# -outdir: set output directory
# -interaction=nonstopmode: don't stop on errors
# -cd: change directory to the source file (useful if building from root)
latexmk -pdf -outdir=$OutputDirectory -interaction=nonstopmode Thesis.tex

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build Successful! PDF is in $OutputDirectory/Thesis.pdf" -ForegroundColor Green
} else {
    Write-Host "Build failed. Check $OutputDirectory/Thesis.log for details." -ForegroundColor Red
}
