# API Documentation

## Endpoints

### GET /
Returns service status.

### GET /health
Returns health check status.

### POST /predict
Accepts JSON payload:

```json
{
  "features": {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MonthlyCharges": 75.2,
    "TotalCharges": 900.5
  }
}
