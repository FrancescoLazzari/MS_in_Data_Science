from bs4 import BeautifulSoup

# Here we define all the functions to parse websites

def extract_MSc_content(file_content):

    """ Function to extract information about the Master degree course, given a file path

    Input:
        file_content (str): It's the path for the .html file downloaded on your computer

    Output:
        tuple: A tuple containing the following information about the Master degree course:
        (
            courseName (str),
            universityName (str),
            facultyName (str),
            isItFullTime (str),
            description (str),
            startDate (str),
            fees (str),
            modality (str),
            duration (str),
            city (str),
            country (str),
            administration (str),
            url (str)
        )
    """

    # Parsing file
    soup = BeautifulSoup(file_content, 'html.parser')

    # Get information about website with try-except block in order to avoid issues
    # For example website https://www.findamasters.com/masters-degrees/course/structural-engineering-msc/?i307d4813c21367 has no informations
    try:
        courseName = soup.find('h1', class_='text-white course-header__course-title').text.strip()
    except:
        courseName=""
    try:
        universityName = soup.find('a', class_='course-header__institution').text.strip()
    except:
        universityName=""
    try:
        facultyName = soup.find('a', class_='course-header__department').text.strip()
    except:
        facultyName=""
    try:
        isItFullTime = soup.find('span', class_='key-info__study-type').text.strip().replace('\n',' ')
    except:
        isItFullTime = ""
    try:
        description = soup.find('div', class_='course-sections__content').text.strip()
    except:
        description=""
    try:
        startDate = soup.find('span', class_='key-info__start-date').text.strip()
    except:
        startDate=""
    try:
        fees = soup.find('div', class_='course-sections__fees')
        fees = fees.find('div', class_='course-sections__content').text.strip()
    except:
        fees=""
    try:
        modality = soup.find('span', class_='key-info__qualification').text.strip()
    except:
        modality=""
    try:
        duration = soup.find('span', class_='key-info__duration').text.strip()
    except:
        duration=""
    try:
        city = soup.find('a', class_='course-data__city').text.strip()
    except:
        city=""
    try:
        country = soup.find('a', class_='course-data__country').text.strip()
    except:
        country=""

    try:
        administration = soup.find('a', class_='course-data__on-campus').text.strip()
    except AttributeError:
        try:
            administration = soup.find('a', class_='course-data__online').text.strip()
        except AttributeError:
            administration = ""
    try:
        url = soup.find('head')
        url = url.find('link', rel='canonical')
        url = url.get('href')
    except:
        url=""


    return(courseName, universityName, facultyName, isItFullTime, description, startDate, fees, modality, duration, city, country, administration, url)