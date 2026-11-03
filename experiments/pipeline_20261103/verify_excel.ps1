$ErrorActionPreference = 'Stop'
$outDir = Join-Path $PSScriptRoot 'outputs/pipeline_20261103'
$path = Join-Path $outDir 'BW1100_分块与搬运优化测试_20261103.xlsx'
$excel = $null
$book = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.EnableEvents = $false
    $book = $excel.Workbooks.Open($path, 0, $false)
    $excel.CalculateFullRebuild()
    foreach ($case in @(@('44组验收',15,34),@('22组独立校准',12,15))) {
        $sheet = $book.Worksheets.Item($case[0])
        if ($sheet.Range('B5').Value2 -ne $case[1] -or $sheet.Range('E5').Value2 -ne $case[2]) { throw 'Pass counts mismatch' }
    }
    if ($book.Worksheets.Item('12组冻结后留出').Range('B5').Value2 -ne 9) { throw 'Holdout mismatch' }
    foreach ($sheet in $book.Worksheets) {
        $errors = $null
        try { $errors = $sheet.UsedRange.SpecialCells(-4123,16) } catch {}
        if ($null -ne $errors) { throw "Formula error in $($sheet.Name)" }
        $area = $sheet.PageSetup.PrintArea
        $tall = $sheet.PageSetup.FitToPagesTall
        if ($sheet.Name -eq '12组冻结后留出') { $sheet.PageSetup.PrintArea = 'A1:L19' } else { $sheet.PageSetup.PrintArea = 'A1:R17' }
        $sheet.PageSetup.FitToPagesTall = 1
        $sheet.ExportAsFixedFormat(0,(Join-Path $outDir ($sheet.Name+'.pdf')))
        $sheet.PageSetup.PrintArea = $area
        $sheet.PageSetup.FitToPagesTall = $tall
    }
    $sheet=$book.Worksheets.Item('44组验收')
    $sheet.Activate()
    $sheet.Range('A1').Select()
    $book.Save()
    Write-Output 'Excel full recalculation: 44/22/12 rows; counts 34/15/9; no formula errors; all three sheets rendered.'
} finally {
    if ($null -ne $book) { $book.Close($true) }
    if ($null -ne $excel) { $excel.Quit() }
    if ($null -ne $book) { [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($book) }
    if ($null -ne $excel) { [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($excel) }
}
