# Share deletion so atomic JSON replacement is not blocked by live monitoring.
$taskProgressPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../artifacts/thucnews_small_search/progress.json'))
$taskShare = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
$taskStream = [System.IO.File]::Open($taskProgressPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, $taskShare)
$taskReader = [System.IO.StreamReader]::new($taskStream, [System.Text.Encoding]::UTF8)
try {
    $taskReader.ReadToEnd()
} finally {
    $taskReader.Dispose()
    $taskStream.Dispose()
}
