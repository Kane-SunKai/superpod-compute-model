$ErrorActionPreference='Stop'
$outDir=Join-Path $PSScriptRoot 'outputs/decode_20261104'
$file=(Get-ChildItem -LiteralPath $outDir -Filter '*.xlsx' | Select-Object -First 1).FullName
$data=Get-Content (Join-Path $PSScriptRoot 'final_analysis.json') -Raw | ConvertFrom-Json
$excel=$null;$book=$null
try {
 $excel=New-Object -ComObject Excel.Application
 $excel.Visible=$false;$excel.DisplayAlerts=$false;$excel.EnableEvents=$false
 $book=$excel.Workbooks.Open($file,0,$false);$excel.CalculateFullRebuild()
 $main=$book.Worksheets.Item(1);$raw=$book.Worksheets.Item(2)
 if($main.Range('B5').Value2 -ne $data.count -or $main.Range('E5').Value2 -ne $data.passed){throw 'Headline count mismatch'}
 for($i=0;$i -lt $data.count;$i++){
  $row=$i+11
  if([math]::Abs($main.Cells.Item($row,6).Value2-$data.cases[$i].measured_us) -gt 1e-7){throw 'Measured mean mismatch'}
  if([math]::Abs($main.Cells.Item($row,8).Value2-$data.cases[$i].signed_error) -gt 1e-9){throw 'Error formula mismatch'}
 }
 $before=$main.Range('F11').Value2;$saved=@()
 for($i=2;$i -le 51;$i++){$saved+=$raw.Cells.Item($i,7).Value2;$raw.Cells.Item($i,7).Value2=$saved[-1]*1.2}
 $excel.CalculateFullRebuild()
 if([math]::Abs($main.Range('F11').Value2-$before*1.2) -gt 1e-7){throw 'Input propagation test failed'}
 for($i=2;$i -le 51;$i++){$raw.Cells.Item($i,7).Value2=$saved[$i-2]}
 $excel.CalculateFullRebuild()
 if([math]::Abs($main.Range('F11').Value2-$before) -gt 1e-7){throw 'Input restore failed'}
 foreach($s in $book.Worksheets){
  $errors=$null;try{$errors=$s.UsedRange.SpecialCells(-4123,16)}catch{}
  if($null -ne $errors){throw 'Formula error found'}
  $area=$s.PageSetup.PrintArea;$tall=$s.PageSetup.FitToPagesTall
  $s.PageSetup.FitToPagesTall=1
  if($s.Index -eq 1){$s.PageSetup.PrintArea='A1:N60'}else{$s.PageSetup.PrintArea='A1:G17'}
  $s.ExportAsFixedFormat(0,(Join-Path $outDir ('sheet'+$s.Index+'.pdf')))
  $s.PageSetup.PrintArea=$area;$s.PageSetup.FitToPagesTall=$tall
 }
 $main.Activate();$main.Range('A1').Select();$book.Save()
 Write-Output 'Verified 36 measured means and errors, 19 passing cases; raw-input propagation and restore; no formula errors; PDF previews saved.'
} finally {
 if($null -ne $book){$book.Close($true)}
 if($null -ne $excel){$excel.Quit()}
 if($null -ne $book){[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($book)}
 if($null -ne $excel){[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($excel)}
}
