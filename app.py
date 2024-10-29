from flask import Flask, render_template, request, redirect, url_for, flash
import requests
import re
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from googletrans import Translator
import nltk

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your secret key

# Initialize stop words for multiple languages
stop_words = {
    'en': set(stopwords.words('english')).union({'said', 'will', 'also', 'one', 'new', 'make'}),
    'id': set(stopwords.words('indonesian')).union({'dan', 'yang', 'di', 'dari', 'pada', 'untuk', 'dengan', 'ke', 'dalam', 'adalah'}),
    'es': set(stopwords.words('spanish')).union({'y', 'el', 'en', 'con', 'para', 'de'}),
    'fr': set(stopwords.words('french')).union({'et', 'le', 'est', 'dans', 'sur', 'avec'})
}

# Fetch article function
def fetch_article(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else 'No Title Found'
        paragraphs = soup.find_all('p')
        article = ' '.join([para.get_text() for para in paragraphs])

        # Clean the article from ads
        ad_patterns = re.compile(r"(Advertisement|Scroll to Continue|Baca Juga|Lanjutkan dengan Konten)", re.IGNORECASE)
        article_cleaned = ad_patterns.sub('', article)
        
        return title, article_cleaned.strip()
    else:
        return None, None

# Summarize article function
def summarize_article_flexible(article, num_clusters=2):
    sentences = sent_tokenize(article)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)

    point_summary = []
    for i in range(num_clusters):
        cluster_sentences = [sentences[j] for j in range(len(sentences)) if kmeans.labels_[j] == i]
        if cluster_sentences:
            point_summary.append(max(cluster_sentences, key=len))  # Longest sentence as key point

    paragraph_summary = ' '.join(point_summary)
    return point_summary, paragraph_summary

# Long summary function
def long_summary(article):
    sentences = sent_tokenize(article)
    return ' '.join(sentences)

# Translate article function
def translate_article(article, dest_language='en'):
    translator = Translator()
    try:
        detected_lang = translator.detect(article).lang
        if detected_lang != dest_language:
            translated = translator.translate(article, dest=dest_language)
            return translated.text
        else:
            return article
    except Exception as e:
        return None

# Hashtag generation function
def generate_hashtags(title, content, lang='en', num_hashtags=5):
    stop_words_set = stop_words.get(lang, set())
    title_words = [word for word in word_tokenize(title.lower()) if word.isalnum() and len(word) > 3 and word not in stop_words_set]
    content_words = [word for word in word_tokenize(content.lower()) if word.isalnum() and len(word) > 3 and word not in stop_words_set]
    
    keywords = title_words * 2 + content_words  # Doubling title words to increase their weight
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([' '.join(keywords)])
    
    tfidf_scores = X.toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    scored_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    
    top_keywords = [f"#{keyword.capitalize()}" for keyword, score in scored_keywords[:num_hashtags]]
    return top_keywords

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        lang = request.form['language']
        
        title, article = fetch_article(url)
        if article:
            translated_title = translate_article(title, lang)
            translated_article = translate_article(article, lang)
            num_clusters = int(request.form['num_clusters'])
            point_summary, paragraph_summary = summarize_article_flexible(translated_article, num_clusters)
            long_paragraph_summary = long_summary(translated_article)  # New long summary
            hashtags = generate_hashtags(translated_title, translated_article, lang)

            return render_template('index.html', title=translated_title, summary=point_summary, paragraph_summary=paragraph_summary, long_summary=long_paragraph_summary, hashtags=hashtags)
        else:
            flash('Failed to fetch article. Please check the URL.')
            return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    app.run(debug=True)
