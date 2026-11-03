$ErrorActionPreference='Stop'
$outDir=Join-Path $PSScriptRoot 'outputs/mapper_library_20261103'
$file=Join-Path $outDir 'BW1100_分块方案库与Mapper独立验证_20261103.xlsx'
$analysis=Get-Content (Join-Path $PSScriptRoot 'analysis.json') -Raw | ConvertFrom-Json
$excel=$null; $book=$null
try {
 $excel=New-Object -ComObject Excel.Application
 $excel.Visible=$false; $excel.DisplayAlerts=$false; $excel.EnableEvents=$false
 $book=$excel.Workbooks.Open($file,0,$false)
 $excel.CalculateFullRebuild()
 $overview=$book.Worksheets.Item('结果总览')
 $expected=@(8,7,14,11,7)
 for($i=0;$i -lt 5;$i++) { if($overview.Cells.Item($i+5,2).Value2 -ne $expected[$i]) { throw 'Summary count mismatch' } }
 $sheet=$book.Worksheets.Item('24组独立验证')
 for($i=0;$i -lt 24;$i++) {
   $actual=$sheet.Cells.Item($i+5,5).Value2
   if([math]::Abs($actual-$analysis.cases[$i].measured_us) -gt 1e-7) { throw 'Raw event to summary mismatch' }
 }
 # Exercise the real raw-source -> median -> error -> summary calculation chain.
 $raw=$book.Worksheets.Item('GPU原始计时');$id=$raw.Range('A5').Value2
 $index=0; while($analysis.cases[$index].id -ne $id){$index++}
 $row=$index+5;$before=$sheet.Cells.Item($row,5).Value2;$saved=@()
 for($i=5;$i -le 54;$i++) {$saved+=$raw.Cells.Item($i,4).Value2;$raw.Cells.Item($i,4).Value2=$saved[-1]*1.2}
 $excel.CalculateFullRebuild()
 if([math]::Abs($sheet.Cells.Item($row,5).Value2-$before*1.2) -gt 1e-7){throw 'Formula dependency test failed'}
 for($i=5;$i -le 54;$i++) {$raw.Cells.Item($i,4).Value2=$saved[$i-5]}
 $excel.CalculateFullRebuild()
 if([math]::Abs($sheet.Cells.Item($row,5).Value2-$before) -gt 1e-7){throw 'Restore failed'}
 foreach($s in $book.Worksheets) {
   $errors=$null;try{$errors=$s.UsedRange.SpecialCells(-4123,16)}catch{}
   if($null -ne $errors){throw "Formula errors: $($s.Name)"}
   $area=$s.PageSetup.PrintArea;$tall=$s.PageSetup.FitToPagesTall
   if($s.Name -eq '结果总览'){$s.PageSetup.PrintArea='A1:G26'}
   elseif($s.Name -eq '24组独立验证'){$s.PageSetup.PrintArea='A1:O28'}
   elseif($s.Name -eq '分块选择'){$s.PageSetup.PrintArea='A1:K17'}
   elseif($s.Name -eq 'GPU原始计时'){$s.PageSetup.PrintArea='A1:E17'}
   elseif($s.Name -eq '12组受控对照'){$s.PageSetup.PrintArea='A1:J17'}
   else{$s.PageSetup.PrintArea='A1:I17'}
   $s.PageSetup.FitToPagesTall=1
   $s.ExportAsFixedFormat(0,(Join-Path $outDir ($s.Name+'.pdf')))
   $s.PageSetup.PrintArea=$area;$s.PageSetup.FitToPagesTall=$tall
 }
 $overview.Activate();$overview.Range('A1').Select();$book.Save()
 Write-Output '7 sheets verified: counts 8/7/14/11/7; 24 actuals match raw medians; input propagation + restore passed; no formula errors; rendered.'
} finally {
 if($null -ne $book){$book.Close($true)}
 if($null -ne $excel){$excel.Quit()}
 if($null -ne $book){[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($book)}
 if($null -ne $excel){[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($excel)}
}
