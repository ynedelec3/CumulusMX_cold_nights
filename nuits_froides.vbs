Dim strCommand, objShell
strCommand = "D:\NedelecDev\git_local\CumulusMX_cold_nights\nuits_froides.lnk"
Set objShell = CreateObject("WScript.Shell")
objShell.Run strCommand,1