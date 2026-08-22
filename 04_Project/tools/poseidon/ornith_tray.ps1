# Ornith/remora server tray monitor for Poseidon (no installs, pure .NET)
# Green dot  = server healthy (port 11435)
# Red dot    = stopped
# Tray menu: Start / Stop / Status / Exit
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$URL  = "http://127.0.0.1:11435/health"
$SRV  = "C:\models\llama-cpp-b10509\llama-server.exe"
$MODEL = "C:\models\Ornith\Ornith-1.5-35B-Q4_K_M.gguf"
$KEY  = "API_KEY_REDACTED"

function Test-Server {
    try {
        $r = Invoke-WebRequest -Uri $URL -UseBasicParsing -TimeoutSec 3
        return ($r.Content -match '"status"\s*:\s*"ok"')
    } catch { return $false }
}

function New-DotIcon([System.Drawing.Color]$color) {
    $bmp = New-Object System.Drawing.Bitmap 16,16
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $brush = New-Object System.Drawing.SolidBrush $color
    $g.FillEllipse($brush, 2, 2, 12, 12)
    $g.Dispose(); $brush.Dispose()
    return [System.Drawing.Icon]::FromHandle($bmp.GetHicon())
}

$tray = New-Object System.Windows.Forms.NotifyIcon
$tray.Visible = $true
$tray.Text = "Ornith server: checking..."

function Update-Icon {
    if (Test-Server) {
        $tray.Icon = New-DotIcon ([System.Drawing.Color]::LimeGreen)
        $tray.Text = "Ornith server: RUNNING (11435)"
    } else {
        $tray.Icon = New-DotIcon ([System.Drawing.Color]::Firebrick)
        $tray.Text = "Ornith server: STOPPED"
    }
}

$mStart = New-Object System.Windows.Forms.ToolStripMenuItem("Start server")
$mStop  = New-Object System.Windows.Forms.ToolStripMenuItem("Stop server")
$mStat  = New-Object System.Windows.Forms.ToolStripMenuItem("Check status")
$mExit  = New-Object System.Windows.Forms.ToolStripMenuItem("Exit tray")

$mStart.Add_Click({
    if (Test-Server) { [System.Windows.Forms.MessageBox]::Show("Already running","Ornith") | Out-Null; return }
    Start-Process -FilePath $SRV -ArgumentList @("-m",$MODEL,"--host","0.0.0.0","--port","11435","-c","65536","-ctk","q8_0","-ctv","q8_0","-ngl","0","--threads","8","--parallel","2","--api-key",$KEY) -WindowStyle Hidden
    Start-Sleep -Seconds 2
    Update-Icon
})
$mStop.Add_Click({
    Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    Update-Icon
})
$mStat.Add_Click({ Update-Icon; [System.Windows.Forms.MessageBox]::Show($tray.Text,"Ornith") | Out-Null })
$mExit.Add_Click({ $tray.Visible = $false; [System.Windows.Forms.Application]::Exit() })

$menu = New-Object System.Windows.Forms.ContextMenuStrip
$menu.Items.Add($mStart) | Out-Null
$menu.Items.Add($mStop) | Out-Null
$menu.Items.Add($mStat) | Out-Null
$menu.Items.Add($mExit) | Out-Null
$tray.ContextMenuStrip = $menu

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 10000
$timer.Add_Tick({ Update-Icon })
$timer.Start()

Update-Icon
[System.Windows.Forms.Application]::Run()
