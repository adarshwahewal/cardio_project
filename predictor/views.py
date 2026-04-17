import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .ml_utils import predict_cardio, get_shap_explanation
from .models import PatientProfile, ScanResult, CustomUser, Prediction


@csrf_exempt
def predict_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    try:
        data = json.loads(request.body)

        required_fields = [
            "patient_name", "age", "gender", "height", "weight",
            "ap_hi", "ap_lo", "cholesterol",
            "gluc", "smoke", "alco", "active"
        ]

        for field in required_fields:
            if data.get(field) is None:
                return JsonResponse({"error": f"{field} is required"}, status=400)

        input_data = {
            "patient_name": data["patient_name"],
            "age":         float(data["age"]),
            "gender":      int(data["gender"]),
            "height":      float(data["height"]),
            "weight":      float(data["weight"]),
            "ap_hi":       float(data["ap_hi"]),
            "ap_lo":       float(data["ap_lo"]),
            "cholesterol": int(data["cholesterol"]),
            "gluc":        int(data["gluc"]),
            "smoke":       int(data["smoke"]),
            "alco":        int(data["alco"]),
            "active":      int(data["active"]),
        }

        # Prediction
        pred, prob = predict_cardio(input_data)
        # SHAP
        shap_contribs, top3 = get_shap_explanation(input_data)
        # Risk label
        if prob >= 0.7:
            risk = "High"
        elif prob >= 0.4:
            risk = "Moderate"
        else:
            risk = "Low"

        # ✅ Save Prediction (existing model)
        prediction_data = dict(input_data)
        prediction_data.pop("patient_name", None)
        Prediction.objects.create(
            result=pred,
            probability=prob,
            **prediction_data
        )

        # ✅ Save ScanResult (dashboard ke liye — yahi missing tha)
        profile, _ = PatientProfile.objects.get_or_create(user=request.user)
        scan = ScanResult.objects.create(
            patient=profile,
            input_data=input_data,
            probability=prob,
            risk_label=risk,
            shap_data=shap_contribs,
            top_reasons=top3,
        )
        return JsonResponse({
            "prediction":  pred,
            "probability": round(prob, 3),
            "risk":        risk,
            "shap":        shap_contribs,
            "top_reasons": top3,
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def register_view(request):
    if request.method == 'POST':
        email    = request.POST.get('email', '').strip()
        username = email.split('@')[0]
        password = request.POST.get('password', '')
        if CustomUser.objects.filter(email=email).exists():
            return render(request, 'auth.html', {'error': 'Email already registered.', 'tab': 'register'})
        user = CustomUser.objects.create_user(
            username=username, email=email, password=password
        )
        login(request, user)
        return redirect('/home/')
    return render(request, 'auth.html', {'tab': 'register'})


def login_view(request):
    if request.method == 'GET' and request.user.is_authenticated:
        return redirect('/home/')
    if request.method == 'POST':
        email    = request.POST.get('email', '')
        password = request.POST.get('password', '')
        user = authenticate(request, username=email, password=password)
        if user:
            login(request, user)
            return redirect('/home/')
        return render(request, 'auth.html', {'error': 'Invalid credentials.', 'tab': 'login'})
    return render(request, 'auth.html', {'tab': 'login'})


def logout_view(request):
    logout(request)
    return redirect('/login/')


@login_required
def home(request):
    return render(request, 'index.html')


@login_required
@ensure_csrf_cookie
def dashboard(request):
    profile, _ = PatientProfile.objects.get_or_create(user=request.user)
    scans = profile.scans.all()[:20]
    scans_data = []
    for s in scans:
        payload = s.input_data or {}
        age_days = payload.get("age")
        age_yr = round(age_days / 365) if age_days else None
        scans_data.append({
            "id": s.id,
            "created_at": s.created_at,
            "probability": s.probability,
            "risk_label": s.risk_label,
            "input_data": payload,
            "bmi": s.bmi(),
            "user_email": profile.user.email,
            "data_json": json.dumps({
                "id": s.id,
                "prob": s.probability,
                "risk": s.risk_label,
                "ageYr": age_yr,
                "payload": payload,
                "createdAt": s.created_at.isoformat(),
            }),
        })
    return render(request, 'dashboard.html', {
        'scans': scans,
        'scans_data': scans_data,
        'profile': profile,
    })


@login_required
def delete_scan(request, scan_id):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    deleted, _ = ScanResult.objects.filter(
        id=scan_id,
        patient__user=request.user,
    ).delete()
    return JsonResponse({"deleted": bool(deleted)})


@login_required
def scan_detail(request, scan_id):
    scan = ScanResult.objects.filter(
        id=scan_id,
        patient__user=request.user,
    ).first()
    if not scan:
        return JsonResponse({"error": "Not found"}, status=404)
    payload = scan.input_data or {}
    age_days = payload.get("age")
    age_yr = round(age_days / 365) if age_days else None
    return JsonResponse({
        "id": scan.id,
        "prob": scan.probability,
        "risk": scan.risk_label,
        "ageYr": age_yr,
        "payload": payload,
        "createdAt": scan.created_at.isoformat(),
    })


@login_required
def scan_history(request):
    profile, _ = PatientProfile.objects.get_or_create(user=request.user)
    scans = profile.scans.all()[:50]
    data = [{
        'date':  s.created_at.strftime('%d %b'),
        'prob':  round(s.probability * 100, 1),
        'label': s.risk_label,
        'bmi':   s.bmi(),
        'bp':    f"{s.input_data.get('ap_hi')}/{s.input_data.get('ap_lo')}",
    } for s in reversed(list(scans))]
    return JsonResponse({'scans': data})


@login_required
def upload_csv(request):
    import pandas as pd
    import csv
    from django.http import HttpResponse

    if request.method == 'POST' and request.FILES.get('csv_file'):
        df       = pd.read_csv(request.FILES['csv_file'])
        required = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                    'cholesterol', 'gluc', 'smoke', 'alco', 'active']

        if not all(c in df.columns for c in required):
            return render(request, 'upload.html', {
                'error': f'Missing columns. Required: {", ".join(required)}'
            })

        results = []
        for _, row in df.iterrows():
            try:
                d = {c: row[c] for c in required}
                _, prob = predict_cardio(d)
                results.append({
                    **d,
                    'probability': round(prob, 3),
                    'risk_label':  'High' if prob >= .7 else 'Moderate' if prob >= .4 else 'Low'
                })
            except Exception as e:
                results.append({**row.to_dict(), 'error': str(e)})

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="cardioscan_results.csv"'
        if results:
            writer = csv.DictWriter(response, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        return response

    return render(request, 'upload.html')