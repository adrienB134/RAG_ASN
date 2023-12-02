import scrapy
from urllib.parse import urljoin


class PdfSpider(scrapy.Spider):
    name = 'asn_scrap'
    start_urls = ['https://www.asn.fr/l-asn-controle/actualites-du-controle/installations-nucleaires/courriers-de-position?page=1']

    def parse(self, response):
        # Extract the PDF URLs from the page
        pdf_urls = response.css('a[href$=".pdf"]::attr(href)').getall()

        # Download the PDF files
        for pdf_url in pdf_urls:
            absolute_pdf_url = urljoin(response.url, pdf_url)
            yield scrapy.Request(absolute_pdf_url, callback=self.save_pdf)
            
        # Check if there are any PDFs on the current page
        if pdf_urls:
            # Get the current page number from the URL
            current_page = int(response.url.split('page=')[-1])

            # Increment the page number
            next_page = current_page + 1

            # Construct the next page URL
            next_page_url = f'https://www.asn.fr/l-asn-controle/actualites-du-controle/installations-nucleaires/courriers-de-position?page={next_page}'

            # Send a request to the next page
            yield scrapy.Request(next_page_url, callback=self.parse)

    def save_pdf(self, response):
        # Save the PDF file to the local filesystem
        path = response.url.split('/')[-1]
        with open(path, 'wb') as f:
            f.write(response.body)