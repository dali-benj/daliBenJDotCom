from django.contrib import admin
from .models import Query


@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    list_display = ("query_text", "timestamp", "ip_address", "user_agent")
    search_fields = ("query_text", "ip_address", "user_agent")
    list_filter = ("timestamp",)
