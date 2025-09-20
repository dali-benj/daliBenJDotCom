# Simple form without reCAPTCHA for debugging
from django import forms

class SimpleQueryForm(forms.Form):
    query = forms.CharField(
        label='Enter your question', 
        max_length=500, 
        widget=forms.TextInput(attrs={'placeholder': 'Ask me anything...'})
    )
