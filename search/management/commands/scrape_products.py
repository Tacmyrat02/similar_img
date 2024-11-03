from django.core.management.base import BaseCommand
from search.models import Product, Category  # Use your actual app name
import requests
from bs4 import BeautifulSoup
from django.core.files.base import ContentFile
from urllib.request import urlopen

class Command(BaseCommand):
    help = 'Scrape products from an external website'

    def handle(self, *args, **kwargs):
        url = "https://gerekli.tm/smartfony-i-aksessuary/smart-chasy/page-3/"
        response = requests.get(url)

        if response.status_code == 200:
            self.stdout.write(self.style.SUCCESS('Successfully retrieved the page.'))
            soup = BeautifulSoup(response.content, "html.parser")
            product_containers = soup.find_all("div", class_="ut2-gl__body")  # Adjust this selector
            
            self.stdout.write(self.style.SUCCESS(f'Found {len(product_containers)} product(s).'))
            category, _ = Category.objects.get_or_create(name='Default Category')

            for product_container in product_containers:
                try:
                    # Extracting the product name
                    title_element = product_container.find("a", class_="product-title")
                    if title_element:
                        product_name = title_element.text.strip()
                    else:
                        self.stdout.write(self.style.ERROR("Product name element not found."))
                        continue

                    # Extracting the product price
                    price_element = product_container.find("span", class_="ty-price-num")
                    if price_element:
                        product_price = price_element.text.strip().replace('₺', '').replace(',', '')
                        price_value = float(product_price)
                    else:
                        self.stdout.write(self.style.ERROR("Product price element not found."))
                        continue

                    # Extracting the product image
                    image_element = product_container.find("img", class_="ty-pict img-ab-hover-gallery")
                    if not image_element:
                        # Check for data-src or other attributes that might hold the image URL
                        image_element = product_container.find("img")
                        if image_element and 'data-src' in image_element.attrs:
                            product_image_url = image_element['data-src']
                        else:
                            self.stdout.write(self.style.ERROR("Product image element not found."))
                            continue
                    else:
                        product_image_url = image_element["src"]

                    # Download the image
                    image_response = urlopen(product_image_url)
                    image_content = image_response.read()
                    image_name = product_image_url.split("/")[-1]
                    product_image_file = ContentFile(image_content)
                    product_image_file.name = image_name

                    # Log product details
                    self.stdout.write(self.style.SUCCESS(f'Product Name: {product_name}, Price: {price_value}, Image URL: {product_image_url}'))

                    product, created = Product.objects.get_or_create(
                        name=product_name,
                        defaults={
                            'price': price_value,
                            'image': product_image_file,
                            'category': category
                        }
                    )

                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"An error occurred: {str(e)}"))
        else:
            self.stdout.write(self.style.ERROR('Failed to retrieve the page.'))
