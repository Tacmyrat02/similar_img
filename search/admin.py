from django.contrib import admin
from .models import Product, Category  # Import Category if you want to add it too

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'image', 'price', 'category')  # Include additional fields
    search_fields = ('name',)
    list_filter = ('category',)  # Enable filtering by category
    prepopulated_fields = {'name': ('name',)}  # Optional: Auto-populate slug field (if you have one)

    # Optional: Add extra customization
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        # You can add extra filtering or annotations here if needed
        return qs

# Optionally register Category if you haven't already
@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    search_fields = ('name',)
