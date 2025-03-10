$headers = @{
    'Content-Type' = 'application/json'
}

$body = @{
    name = "Test Restaurant"
    location = "Indiranagar"
    rest_type = "Fine Dining"
    cuisines = "North Indian, Continental"
    cost_for_two = "2500"
    online_order = "No"
    book_table = "Yes"
    votes = 1000
} | ConvertTo-Json

Write-Host "Testing single prediction endpoint..."
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Headers $headers -Body $body
    Write-Host "Prediction successful!"
    Write-Host "Restaurant: $($response.restaurant_name)"
    Write-Host "Predicted Rating: $($response.predicted_rating)"
} catch {
    Write-Host "Error occurred: $_"
}

# Test batch prediction
$batchBody = @(
    @{
        name = "Restaurant 1"
        location = "Indiranagar"
        rest_type = "Fine Dining"
        cuisines = "North Indian"
        cost_for_two = "2500"
        online_order = "No"
        book_table = "Yes"
        votes = 1000
    },
    @{
        name = "Restaurant 2"
        location = "Koramangala"
        rest_type = "Casual Dining"
        cuisines = "South Indian"
        cost_for_two = "800"
        online_order = "Yes"
        book_table = "No"
        votes = 500
    }
) | ConvertTo-Json

Write-Host "`nTesting batch prediction endpoint..."
try {
    $batchResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/batch" -Method Post -Headers $headers -Body $batchBody
    Write-Host "Batch prediction successful!"
    foreach ($prediction in $batchResponse) {
        Write-Host "Restaurant: $($prediction.restaurant_name)"
        Write-Host "Predicted Rating: $($prediction.predicted_rating)"
    }
} catch {
    Write-Host "Error occurred: $_"
} 