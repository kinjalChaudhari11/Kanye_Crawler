import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import time
import csv

class WikiCrawler:
    def __init__(self, start_url, search_terms=None):
        self.start_url = start_url
        self.visited_urls = set()
        self.downloaded_count = 0
        self.articles = []
        self.search_terms = search_terms or []
    
    def is_valid_wikipedia_url(self, url):
        parsed = urlparse(url)
        return (
            parsed.netloc == 'en.wikipedia.org' and
            '/wiki/' in url and
            '#' not in url and
            ':' not in url.split('/wiki/')[-1] and
            'Main_Page' not in url
        )
    
    def extract_text_from_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        #cleaning search taking inspo and guadance from beautiful soup reference docs 
        for unwanted in soup.find_all(['script', 'style', 'footer', 'header', 'nav']):
            unwanted.decompose()
        
        content_div = soup.find(id='mw-content-text')
        if content_div:
            for ref in content_div.find_all('div', class_='reflist'):
                ref.decompose()
            
            text = content_div.get_text(separator=' ', strip=True)
            return text
        return ""
    
    def crawl(self, max_pages=10):
        queue = [self.start_url]
        
        while queue and self.downloaded_count < max_pages:
            current_url = queue.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            try:
                response = requests.get(current_url)
                
                self.visited_urls.add(current_url)
                text_content = self.extract_text_from_html(response.text)
                
                if text_content:
                    self.articles.append([
                        str(self.downloaded_count + 1),
                        current_url.split('/wiki/')[-1],
                        text_content
                    ])
                    
                    self.downloaded_count += 1
                    #check it should show whats being downloaded now 
                    print(f"Downloaded article {self.downloaded_count} from {current_url}")
                
                    if self.downloaded_count < max_pages:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        for link in soup.find_all('a'):
                            href = link.get('href')
                            if href:
                                full_url = urljoin(current_url, href)
                                if (self.is_valid_wikipedia_url(full_url) and 
                                    full_url not in self.visited_urls and
                                    (not self.search_terms or 
                                     any(term in full_url.lower() for term in self.search_terms))):
                                    queue.append(full_url)
                #debugging
            except Exception as e:
                print(f"Error processing {current_url}: {str(e)}")
                continue
    
    def save_to_csv(self, filename='crawler.csv'): 
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.articles)
        print(f"\nSaved to {filename}")
    
def main():
    start_url = "https://en.wikipedia.org/wiki/Kanye_West"
                                                                            ## pablo is in the name of a popular Album lol funny coincidence 
    search_terms = ['rap', 'hip', 'hop', 'music', 'album', 'kanye', 'west', 'pablo', 'donda','yeezy']
    
    crawler = WikiCrawler(start_url, search_terms)
    crawler.crawl(max_pages=10)
    crawler.save_to_csv('crawler.csv')  
    

if __name__ == "__main__":
    main()