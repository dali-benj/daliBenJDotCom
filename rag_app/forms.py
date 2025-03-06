# rag_app/forms.py
from django import forms
from django_recaptcha.fields import ReCaptchaField
from django_recaptcha.widgets import ReCaptchaV2Checkbox, ReCaptchaV3

class QueryForm(forms.Form):
    query = forms.CharField(label='Enter your question', max_length=500, widget=forms.TextInput(attrs={'placeholder': 'Ask me anything...'}))
    captcha = ReCaptchaField(widget=ReCaptchaV3) # Or ReCaptchaV3 for invisible