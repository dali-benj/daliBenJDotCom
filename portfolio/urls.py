from django.urls import path
from django.views.generic import TemplateView
from . import views

urlpatterns = [
    path('', views.home_view, name='portfolio_home'),
    path('portfolio/otakuthon/', TemplateView.as_view(template_name='portfolio/otakuthon_portfolio.html'), name='otakuthon_portfolio'),
]
