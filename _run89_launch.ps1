Set-Location C:\Users\darf3\Documents\intelligent_data_detective
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:OPENAI_API_KEY = [System.Environment]::GetEnvironmentVariable('OPENAI_API_KEY','User')
$tu = [System.Environment]::GetEnvironmentVariable('TAVILY_API_KEY','User')
if ($tu) { $env:TAVILY_API_KEY = $tu }
"=== RUN 89 LAUNCH $(Get-Date -Format o) PID=$PID ===" | Out-File -Encoding utf8 -Append notebook_run_log.txt
python run_notebook_live.py *>> run89_stdout.log
"=== RUN 89 EXIT $(Get-Date -Format o) code=$LASTEXITCODE ===" | Out-File -Encoding utf8 -Append notebook_run_log.txt
