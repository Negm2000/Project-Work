# latexmk configuration file
# This ensures that all build outputs go to the 'Report' directory

$out_dir = 'Report';
$pdf_mode = 1; # Generate PDF
$bibtex_use = 2; # Use biber if mentioned in source
$clean_ext = 'run.xml bcf synctex.gz fdb_latexmk fls';
