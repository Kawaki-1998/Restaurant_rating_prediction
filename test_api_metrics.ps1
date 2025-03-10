Write-Host "Testing model metrics endpoint..."
try {
    $metrics = Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/metrics" -Method Get
    Write-Host "Model Metrics:"
    Write-Host "R2 Score: $($metrics.r2_score)"
    Write-Host "Model Type: $($metrics.model_type)"
    Write-Host "Number of Features: $($metrics.feature_count)"
    Write-Host "`nModel Parameters:"
    $metrics.parameters.PSObject.Properties | ForEach-Object {
        Write-Host "$($_.Name): $($_.Value)"
    }
} catch {
    Write-Host "Error occurred: $_"
}

Write-Host "`nTesting feature importance endpoint..."
try {
    $importance = Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/feature_importance" -Method Get
    Write-Host "Top 5 Most Important Features:"
    $importance | Select-Object -First 5 | ForEach-Object {
        Write-Host "$($_.feature_name): $($_.importance_score)"
    }
} catch {
    Write-Host "Error occurred: $_"
}

Write-Host "`nTesting feature importance plot endpoint..."
try {
    $plot = Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/feature_importance_plot" -Method Get
    $plotPath = "feature_importance_plot.png"
    [System.Convert]::FromBase64String($plot.image) | Set-Content $plotPath -Encoding Byte
    Write-Host "Feature importance plot saved as: $plotPath"
} catch {
    Write-Host "Error occurred: $_"
} 