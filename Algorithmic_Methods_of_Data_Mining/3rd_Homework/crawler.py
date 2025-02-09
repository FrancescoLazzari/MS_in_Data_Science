import requests
from bs4 import BeautifulSoup

# Here we define all the functions to crawl websites

def extract_MSc(url):

    """ Function to extract names of Master's degrees from a given URL

    Input: 
        url (str): The URL of the page containing Master's degree information.

    Output:
        master_degree_urls (list): A list of URLs pointing to Master's degree programs on the given page.
    """

    r = requests.get(url)
    r.raise_for_status() # raise exception for bad responses

    soup = BeautifulSoup(r.content,'html.parser')
    a = [a['href'] for a in soup.find_all('a', class_='courseLink text-dark')]

    return a



