from django.db import models

class Prediction(models.Model):
    age         = models.FloatField()
    gender      = models.IntegerField()
    height      = models.FloatField()
    weight      = models.FloatField()
    ap_hi       = models.FloatField()
    ap_lo       = models.FloatField()
    cholesterol = models.IntegerField()
    gluc        = models.IntegerField()
    smoke       = models.IntegerField()
    alco        = models.IntegerField()
    active      = models.IntegerField()

    result      = models.IntegerField(blank=True, null=True)
    probability = models.FloatField(blank=True, null=True)

    # ✅ NEW — stores top-5 SHAP contributions as JSON
    shap_data   = models.JSONField(blank=True, null=True)
    # ✅ NEW — stores top-3 readable reasons
    top_reasons = models.JSONField(blank=True, null=True)

    created_at  = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.id} — {self.result}"

    def save(self, *args, **kwargs):
        from .ml_utils import predict_cardio, get_shap_explanation

        input_data = {
            "age": self.age, "gender": self.gender,
            "height": self.height, "weight": self.weight,
            "ap_hi": self.ap_hi, "ap_lo": self.ap_lo,
            "cholesterol": self.cholesterol, "gluc": self.gluc,
            "smoke": self.smoke, "alco": self.alco,
            "active": self.active,
        }

        pred, prob          = predict_cardio(input_data)
        shap_contribs, top3 = get_shap_explanation(input_data)

        self.result      = pred
        self.probability = prob
        self.shap_data   = shap_contribs   # list of dicts → JSONField
        self.top_reasons = top3            # list of strings → JSONField

        super().save(*args, **kwargs)


from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    # ✅ Yeh do lines add karo — yahi fix hai
    groups = models.ManyToManyField(
        'auth.Group',
        blank=True,
        related_name='customuser_set',       # clashing name fix
        related_query_name='customuser',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        blank=True,
        related_name='customuser_set',       # clashing name fix
        related_query_name='customuser',
    )

    def __str__(self):
        return self.email



class PatientProfile(models.Model):
    user       = models.OneToOneField(
                     CustomUser, on_delete=models.CASCADE,
                     related_name='profile')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Profile — {self.user.email}"

class ScanResult(models.Model):
    patient     = models.ForeignKey(
                      PatientProfile, on_delete=models.CASCADE,
                      related_name='scans')
    input_data  = models.JSONField()
    probability = models.FloatField()
    risk_label  = models.CharField(max_length=20)
    shap_data   = models.JSONField(null=True, blank=True)
    top_reasons = models.JSONField(null=True, blank=True)
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def bmi(self):
        h = self.input_data.get('height', 1)
        w = self.input_data.get('weight', 0)
        return round(w / ((h/100)**2), 1)