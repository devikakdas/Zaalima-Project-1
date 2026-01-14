$Url = "http://localhost:5000/predict"

$Body = @{
  robot_id = "ARM_247"
  sensor_readings = @{
    vibration   = @(0.65, 0.67, 0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.74, 0.72, 0.75)
    temperature = @(85, 86, 88, 87, 89, 88, 90, 89, 91, 90, 92, 91)
    pressure    = @(170, 172, 175, 173, 178, 176, 180, 177, 182, 179, 185, 183)
  }
  timestamp = "2026-01-08T13:30:00Z"
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri $Url -Method Post -Body $Body -ContentType "application/json"