# NEED CATHSOLID.txt in same directory

$TRAIN_RATIO=0.7
$VAL_RATIO=0.15

$data=Get-Content CATHSOLID.txt | ForEach-Object {
    $p=$_ -split "\s+"
    [PSCustomObject]@{
        ID=$p[0]
        Class=[int]$p[1]
        PDB=$p[0].Substring(0,4).ToLower()}}

$data=$data | Where-Object {$_.Class -ne 6}

$train=@()
$val=@()
$test=@()

foreach ($class in 1..4) {

    $classData=$data | Where-Object {$_.Class -eq $class}

    $shuffled=$classData | Get-Random -Count $classData.Count

    $total=$shuffled.Count
    $trainCount=[int]($total*$TRAIN_RATIO)
    $valCount=[int]($total*$VAL_RATIO)

    $train+=$shuffled[0..($trainCount-1)]
    $val+=$shuffled[$trainCount..($trainCount+$valCount-1)]
    $test+=$shuffled[($trainCount+$valCount)..($total-1)]
}

$train=$train | Get-Random -Count $train.Count
$val =$val | Get-Random -Count $val.Count
$test=$test | Get-Random -Count $test.Count

$train | Export-Csv train.csv -NoTypeInformation
$val | Export-Csv val.csv   -NoTypeInformation
$test  | Export-Csv test.csv  -NoTypeInformation

Write-Host "Splits created."
