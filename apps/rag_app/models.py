from django.db import models
from django.utils import timezone

class Query(models.Model):
    query_text = models.CharField(max_length=500)
    timestamp = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=255, null=True, blank=True)
    answer_text = models.TextField(null=True, blank=True)  # Add this field

    def __str__(self):
        return self.query_text