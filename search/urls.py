from django.urls import path
from .views import image_search, product_list, home, add_product, product_detail,register,user_login,user_logout

urlpatterns = [
    path('', home, name='home'), 
    path('image_search/', image_search, name='image_search'),
    path('add_product/', add_product, name='add_product'),
    path('products/', product_list, name='product_list'),
    path('product/<int:product_id>/', product_detail, name='product_detail'),
    path('register/', register, name='register'),
    path('login/', user_login, name='login'),
    path('logout/', user_logout, name='logout'),
]
